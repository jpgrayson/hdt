// //! *This module is available only if HDT is built with the experimental `"nt"` feature.*
use super::concurrent_interner::{Interner, Terms};
use crate::containers::rdf::Id;
use crate::header::Header;
use crate::triples::{Id as HdtId, TripleId, TriplesBitmap};
use crate::{DictSectPFC, FourSectDict, Hdt};
use bitset_core::BitSet;
use bytesize::ByteSize;
use log::{debug, error};
use oxrdf::{Term, vocab::xsd};
use oxttl::NTriplesParser;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::io::{Error, ErrorKind::InvalidData};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;

pub type Result<T> = std::io::Result<T>;
type Simd = [u64; 4];
type Indices = Vec<Simd>;

/// What to do with a triple that has a nul char (U+0000) in the value of a literal.
///
/// N-Triples permits U+0000 in literals, written `\u0000`, `\U00000000` or as a raw byte,
/// but HDT cannot store it: dictionary entries are nul-terminated, so a nul would end the
/// entry early and corrupt its front-coded neighbours. No policy is therefore lossless.
/// hdt-cpp and rapper truncate such values silently; this crate errors unless told otherwise.
///
/// A nul char anywhere else, in an IRI, a blank node label, a language tag or a datatype, is
/// not valid RDF and is always an error, whatever the policy: rewriting it would make a
/// different identifier.
///
/// The lossy policies rewrite or drop triples before the dictionary is built, so terms that
/// become equal are merged like any other duplicate. [`NtReport`] counts the affected triples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum NulPolicy {
    /// Fail with an [`std::io::ErrorKind::InvalidData`] error showing the offending triple.
    #[default]
    Reject,
    /// Drop the whole triple. Lossy.
    Drop,
    /// Cut each affected literal value at its first nul, keeping the closing quote and the
    /// language tag or datatype, as hdt-cpp and rapper do. Lossy.
    Truncate,
    /// Remove every nul char from each affected literal value. Lossy.
    Strip,
}

/// Options for [`Hdt::read_nt_with`] and [`Hdt::from_triples_with`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NtOptions {
    /// How to handle nul chars (U+0000) in terms, [`NulPolicy::Reject`] by default.
    pub nul_policy: NulPolicy,
}

impl NtOptions {
    /// Set the [`NulPolicy`].
    #[must_use]
    pub const fn nul_policy(mut self, nul_policy: NulPolicy) -> Self {
        self.nul_policy = nul_policy;
        self
    }
}

/// What [`Hdt::read_nt_with`] and [`Hdt::from_triples_with`] did to the input beyond converting it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct NtReport {
    /// Number of input triples with a nul char in a literal, dropped or rewritten according to the [`NulPolicy`].
    pub nul_triples: usize,
}

impl Hdt {
    /// Converts RDF N-Triples to HDT with a FourSectionDictionary with DictionarySectionPlainFrontCoding and SPO order.
    /// Literal escapes are decoded, so the dictionary holds term values, not N-Triples syntax.
    /// Fails if a literal contains a nul char (U+0000), which HDT cannot store; see [`NulPolicy`] and [`Hdt::read_nt_with`].
    /// *This function is available only if HDT is built with the experimental `"nt"` feature.*
    /// # Example
    /// ```
    /// let hdt = hdt::Hdt::read_nt("tests/resources/empty.nt").unwrap();
    /// ```
    pub fn read_nt(f: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::read_nt_with(f, &NtOptions::default())?.0)
    }

    /// Like [`Hdt::read_nt`], with [`NtOptions`] and an [`NtReport`] on what was dropped or rewritten.
    /// *This function is available only if HDT is built with the experimental `"nt"` feature.*
    /// # Example
    /// ```
    /// use hdt::{Hdt, NtOptions, NulPolicy};
    /// let options = NtOptions::default().nul_policy(NulPolicy::Drop);
    /// let (hdt, report) = Hdt::read_nt_with("tests/resources/empty.nt", &options).unwrap();
    /// assert_eq!(report.nul_triples, 0);
    /// ```
    pub fn read_nt_with(f: impl AsRef<Path>, options: &NtOptions) -> Result<(Self, NtReport)> {
        let f = f.as_ref();
        let base = Id::Named(format!("file://{}", f.canonicalize()?.display()));
        let original_size = std::fs::File::open(f)?.metadata()?.len();
        let (pool, report) = parse_nt_terms(f, options.nul_policy)?;
        Ok((Self::from_parsed_terms(pool, &base, Some(original_size))?, report))
    }

    /// Builds an HDT with a FourSectionDictionary with DictionarySectionPlainFrontCoding and SPO order
    /// from triples in memory, e.g. to write an existing RDF graph as HDT without going through a file.
    /// Terms are given in the HDT dictionary string format: IRIs without enclosing angle brackets,
    /// literals including quotes, e.g. `"example"@en` or `"123"^^<http://www.w3.org/2001/XMLSchema#integer>`,
    /// and blank nodes as `_:b1`. This is the same format that [`Hdt::triples_all`] returns.
    /// The base IRI denotes the dataset in the header.
    /// Fails if a literal contains a nul char (U+0000), which HDT cannot store; see [`NulPolicy`] and [`Hdt::from_triples_with`].
    /// A nul char in any other term is always an error.
    /// *This function is available only if HDT is built with the experimental `"nt"` feature.*
    /// # Example
    /// ```
    /// let triples = [["http://example.org/subject", "http://example.org/predicate", "\"object\"@en"]];
    /// let hdt = hdt::Hdt::from_triples(triples, "http://example.org/mydataset").unwrap();
    /// ```
    pub fn from_triples<S: AsRef<str>>(triples: impl IntoIterator<Item = [S; 3]>, base_iri: &str) -> Result<Self> {
        Ok(Self::from_triples_with(triples, base_iri, &NtOptions::default())?.0)
    }

    /// Like [`Hdt::from_triples`], with [`NtOptions`] and an [`NtReport`] on what was dropped or rewritten.
    /// *This function is available only if HDT is built with the experimental `"nt"` feature.*
    /// # Example
    /// ```
    /// use hdt::{Hdt, NtOptions, NulPolicy};
    /// let triples = [["http://example.org/s", "http://example.org/p", "\"a\0b\""]];
    /// let options = NtOptions::default().nul_policy(NulPolicy::Strip);
    /// let (hdt, report) = Hdt::from_triples_with(triples, "http://example.org/mydataset", &options).unwrap();
    /// assert_eq!(report.nul_triples, 1);
    /// assert_eq!(hdt.triples_with_pattern(Some("http://example.org/s"), None, Some("\"ab\"")).count(), 1);
    /// ```
    pub fn from_triples_with<S: AsRef<str>>(
        triples: impl IntoIterator<Item = [S; 3]>, base_iri: &str, options: &NtOptions,
    ) -> Result<(Self, NtReport)> {
        let (pool, report) = intern_terms(triples, options.nul_policy)?;
        Ok((Self::from_parsed_terms(pool, &Id::Named(base_iri.to_owned()), None)?, report))
    }

    fn from_parsed_terms(pool: ParsedTerms, base: &Id, original_size: Option<u64>) -> Result<Self> {
        const BLOCK_SIZE: usize = 16;

        let (dict, mut encoded_triples) = dict_triples(pool, BLOCK_SIZE)?;
        let num_triples = encoded_triples.len();
        // Sort by final HDT ID (SPO order) before feeding into TriplesBitmap.
        encoded_triples.par_sort_unstable();
        // Move (don't borrow) the encoded triples in so TriplesBitmap can free
        // them after its build loop, before its op-index peak.
        let triples = TriplesBitmap::from_triples(encoded_triples);

        let header = Header { format: "ntriples".to_owned(), length: 0, body: BTreeSet::new() };
        let mut hdt = Hdt { header, dict, triples };
        hdt.fill_header(base, BLOCK_SIZE, num_triples, original_size);

        debug!("HDT size in memory {}, details:", ByteSize(hdt.size_in_bytes() as u64));
        debug!("{hdt:#?}");
        Ok(hdt)
    }

    /// Populate HDT header fields.
    /// Some fields may be optional, populating same triples as those in C++ version for now.
    fn fill_header(&mut self, base: &Id, block_size: usize, num_triples: usize, original_size: Option<u64>) {
        use crate::containers::rdf::Term::Literal as Lit;
        use crate::containers::rdf::{Literal, Term, Triple};
        use crate::vocab::*;

        const ORDER: &str = "SPO";

        macro_rules! literal {
            ($s:expr, $p:expr, $o:expr) => {
                self.header.body.insert(Triple::new($s.clone(), $p.to_owned(), Lit(Literal::new($o.to_string()))));
            };
        }
        macro_rules! insert_id {
            ($s:expr, $p:expr, $o:expr) => {
                self.header.body.insert(Triple::new($s.clone(), $p.to_owned(), Term::Id($o.clone())));
            };
        }
        literal!(base, RDF_TYPE, HDT_CONTAINER);
        literal!(base, RDF_TYPE, VOID_DATASET);
        literal!(base, VOID_TRIPLES, num_triples);
        literal!(base, VOID_PROPERTIES, self.dict.predicates.num_strings);
        let [d_s, d_o] =
            [&self.dict.subjects, &self.dict.objects].map(|s| s.num_strings + self.dict.shared.num_strings);
        literal!(base, VOID_DISTINCT_SUBJECTS, d_s);
        literal!(base, VOID_DISTINCT_OBJECTS, d_o);
        // // TODO: Add more VOID Properties. E.g. void:classes

        // // Structure
        let stats_id = Id::Blank("statistics".to_owned());
        let pub_id = Id::Blank("publicationInformation".to_owned());
        let format_id = Id::Blank("format".to_owned());
        let dict_id = Id::Blank("dictionary".to_owned());
        let triples_id = Id::Blank("triples".to_owned());
        insert_id!(base, HDT_STATISTICAL_INFORMATION, stats_id);
        insert_id!(base, HDT_STATISTICAL_INFORMATION, pub_id);
        insert_id!(base, HDT_FORMAT_INFORMATION, format_id);
        insert_id!(format_id, HDT_DICTIONARY, dict_id);
        insert_id!(format_id, HDT_TRIPLES, triples_id);
        // DICTIONARY
        literal!(dict_id, HDT_DICT_SHARED_SO, self.dict.shared.num_strings);
        literal!(dict_id, HDT_DICT_MAPPING, "1");
        literal!(dict_id, HDT_DICT_SIZE_STRINGS, ByteSize(self.dict.size_in_bytes() as u64));
        literal!(dict_id, HDT_DICT_BLOCK_SIZE, block_size);
        // TRIPLES
        literal!(triples_id, DC_TERMS_FORMAT, HDT_TYPE_BITMAP);
        literal!(triples_id, HDT_NUM_TRIPLES, num_triples);
        literal!(triples_id, HDT_TRIPLES_ORDER, ORDER);
        // // Sizes
        if let Some(size) = original_size {
            literal!(stats_id, HDT_ORIGINAL_SIZE, size);
        }
        // a few bytes off because that literal itself is not counted
        literal!(stats_id, HDT_SIZE, ByteSize(self.size_in_bytes() as u64));
        // exclude for now to skip dependency on chrono
        //let datetime_str = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%z").to_string();
        //literal!(pub_id,DC_TERMS_ISSUED,datetime_str);
    }
}

/// Output of [`parse_nt_terms`] (file path) and [`intern_terms`] (in-memory).
/// All term strings live inside the `Interner`; the triples hold `u32` term
/// indices (4 bytes each) instead of full strings, and the three bitsets track
/// which indices appear as subject / predicate / object.
struct ParsedTerms {
    triples: Vec<[u32; 3]>,
    interner: Interner,
    subjects: Indices,
    predicates: Indices,
    objects: Indices,
}

impl ParsedTerms {
    /// Derive the role bitsets (subject / predicate / object) from the interned
    /// triples. Indices are 0-based and dense, sized by the interner's term count.
    fn new(interner: Interner, triples: Vec<[u32; 3]>) -> Self {
        let block = [0u64; 4];
        let blocks = interner.len().div_ceil(256);
        let mut subjects: Indices = vec![block; blocks];
        let mut objects: Indices = vec![block; blocks];
        let mut predicates: Indices = vec![block; blocks];

        for [s, p, o] in &triples {
            subjects.bit_set(*s as usize);
            predicates.bit_set(*p as usize);
            objects.bit_set(*o as usize);
        }

        ParsedTerms { triples, interner, subjects, predicates, objects }
    }
}

/// ID map: indexed by term index (`u32` as `usize`), holds the final HDT id for
/// a term in a given role (subject/predicate/object), or 0 if it has no id in
/// that role. u32 fits: HDT ids are at most `num_strings` ≤ u32::MAX.
type IdMap = Vec<u32>;

/// Intern one triple, applying `policy` if a literal value contains a nul char.
///
/// This must happen before interning: the lossy policies can make distinct terms equal,
/// which sorting, deduplication and section assignment only handle if they never see the
/// original. Returns `None` for a dropped triple. A clean triple, the overwhelmingly
/// common case, costs one memchr per term and no allocation.
fn intern_triple(
    interner: &Interner, t: [&str; 3], policy: NulPolicy, nul_triples: &AtomicUsize,
) -> Option<Result<[u32; 3]>> {
    if !t.iter().any(|term| term.contains('\0')) {
        return Some(Ok(t.map(|term| interner.get_or_intern(term))));
    }
    nul_triples.fetch_add(1, Relaxed);
    // a nul outside a literal value is invalid RDF, not a limit of HDT, so no policy applies to it
    if let Some(i) = t.iter().position(|term| term.contains('\0') && !nul_only_in_literal_value(term)) {
        return Some(Err(nul_error(
            &t, i, "which is not valid in an IRI, blank node label, language tag or datatype",
        )));
    }
    if policy == NulPolicy::Reject {
        let i = t.iter().position(|term| term.contains('\0')).expect("a term has a nul");
        return Some(Err(nul_error(&t, i, "which HDT cannot store")));
    }
    let t: [String; 3] = match policy {
        NulPolicy::Reject => unreachable!("rejected above"),
        NulPolicy::Drop => return None,
        NulPolicy::Truncate => t.map(truncate_nul),
        NulPolicy::Strip => t.map(|term| term.replace('\0', "")),
    };
    Some(Ok(t.each_ref().map(|term| interner.get_or_intern(term))))
}

/// Whether every nul char of `term`, in dictionary form, lies inside a literal's quotes.
fn nul_only_in_literal_value(term: &str) -> bool {
    term.starts_with('"')
        && term.rfind('"').zip(term.rfind('\0')).is_some_and(|(closing_quote, last_nul)| last_nul < closing_quote)
}

/// An error showing the whole triple, so that it can be found in the input.
fn nul_error(t: &[&str; 3], i: usize, what: &str) -> Error {
    let show = |term: &str| {
        let term =
            if term.starts_with('"') || term.starts_with("_:") { term.to_owned() } else { format!("<{term}>") };
        term.replace('\0', "\\u0000")
    };
    let role = ["subject", "predicate", "object"][i];
    Error::new(
        InvalidData,
        format!("{} {} {} has a nul char (U+0000) in the {role}, {what}", show(t[0]), show(t[1]), show(t[2])),
    )
}

/// Cut a literal at the first nul of its value, keeping the closing quote and any language
/// tag or datatype, in dictionary form.
fn truncate_nul(term: &str) -> String {
    let Some(nul) = term.find('\0') else { return term.to_owned() };
    let suffix = term.rfind('"').filter(|&q| q > nul).map_or("", |q| &term[q..]);
    [&term[..nul], suffix].concat()
}

/// Intern in-memory string triples into a [`ParsedTerms`]. Single-threaded — the
/// input is one sequential iterator, so there is no parser-level parallelism to
/// exploit here (dictionary compression below still runs on four threads).
fn intern_terms<S: AsRef<str>>(
    triples: impl IntoIterator<Item = [S; 3]>, policy: NulPolicy,
) -> Result<(ParsedTerms, NtReport)> {
    let interner = Interner::new();
    let nul_triples = AtomicUsize::new(0);
    let triples: Vec<[u32; 3]> = triples
        .into_iter()
        .enumerate()
        .filter_map(|(i, t)| {
            intern_triple(&interner, t.each_ref().map(AsRef::as_ref), policy, &nul_triples)
                .map(|r| r.map_err(|e| Error::new(InvalidData, format!("triple {i}: {e}"))))
        })
        .collect::<Result<_>>()?;
    Ok((ParsedTerms::new(interner, triples), NtReport { nul_triples: nul_triples.into_inner() }))
}

/// Convert a parsed/interned term pool to a dictionary and encoded triple IDs.
fn dict_triples(pool: ParsedTerms, block_size: usize) -> Result<(FourSectDict, Vec<TripleId>)> {
    let ParsedTerms { triples, interner, subjects, predicates, objects } = pool;

    // In parallel with dictionary build: sort + dedup triples (by term index
    // — this removes exact duplicate triples; the final SPO-ID sort happens
    // later, once we've assigned HDT ids).
    let sorter = thread::Builder::new().name("sorter".to_owned()).spawn(move || {
        let mut t = triples;
        t.par_sort_unstable();
        t.dedup();
        t
    })?;

    // Assign HDT ids in sorted-string order and build the compressed dict.
    // Returns three `index -> u32 id` lookup tables — direct array indexing
    // during encoding, no more binary-search-through-PFC.
    let (dict, subj_map, pred_map, obj_map) = {
        // Consume the interner into an arena-backed, index-addressable view (no
        // per-term copy), then drop it at the end of this block so the term
        // bytes are freed before the encoding peak.
        let terms = interner.into_terms();
        build_dict_and_id_maps(&terms, &subjects, &predicates, &objects, block_size)
    };
    // Bitsets served their purpose; drop before the encoding peak.
    drop(subjects);
    drop(predicates);
    drop(objects);

    // Drain the sorted index triples directly into HDT-id triples via the ID
    // maps. `into_par_iter` consumes the Vec so the index triples are freed
    // before this function returns — only `Vec<TripleId>` survives into
    // `TriplesBitmap::from_triples`.
    let sorted_triples = sorter.join().expect("NT sorter thread panicked");
    let encoded_triples: Vec<TripleId> = sorted_triples
        .into_par_iter()
        .map(|[s_idx, p_idx, o_idx]| {
            let s = subj_map[s_idx as usize] as HdtId;
            let p = pred_map[p_idx as usize] as HdtId;
            let o = obj_map[o_idx as usize] as HdtId;
            if s == 0 || p == 0 || o == 0 {
                error!("encoded triple [{s}, {p}, {o}] contains 0; term missing from dictionary");
            }
            [s, p, o]
        })
        .collect();

    drop(subj_map);
    drop(pred_map);
    drop(obj_map);

    Ok((dict, encoded_triples))
}

/// Render a parsed term in HDT dictionary form: an IRI unbracketed, a blank
/// node as `_:id`, a literal quoted with its language tag or datatype.
///
/// A literal's body is its *decoded* value. N-Triples escapes are syntax,
/// not part of the value: `"tab\there"` in a file denotes a string holding
/// one tab, and that tab is what belongs in the dictionary. Rendering the
/// term with `to_string()` instead put the escape back in, so serializing
/// the HDT escaped it a second time and a tab came back out as a backslash
/// (#131).
///
/// The HDT format itself never pins this down, but decoded values are the
/// de facto convention: hdt-cpp (through serd) and hdt-java (through Jena)
/// both store them and escape only on serialization, as do
/// [`Hdt::from_triples`] and the Sophia adapter in this crate.
fn dict_string(term: &Term) -> String {
    match term {
        Term::NamedNode(n) => n.as_str().to_owned(),
        Term::BlankNode(b) => format!("_:{}", b.as_str()),
        Term::Literal(l) => {
            let value = l.value();
            match l.language() {
                Some(language) => format!("\"{value}\"@{language}"),
                // A simple literal is an xsd:string literal; both spell the
                // same term and the bare form is canonical, which is what
                // this path has always written.
                None if l.datatype() == xsd::STRING => format!("\"{value}\""),
                None => format!("\"{value}\"^^<{}>", l.datatype().as_str()),
            }
        }
        // RDF-star quoted triples: oxrdf only builds these with its rdf-12
        // feature, which is not enabled here.
        #[allow(unreachable_patterns)]
        other => other.to_string(),
    }
}

/// Parse N-Triples in parallel and collect terms into the interning pool + role bitsets.
fn parse_nt_terms(path: &Path, policy: NulPolicy) -> Result<(ParsedTerms, NtReport)> {
    let interner: Arc<Interner> = Arc::new(Interner::new());
    let nul_triples = AtomicUsize::new(0);
    // use two threads when available parallelism cannot be determined as going to a single thread is around 38% slower
    // 16 chosen as a sane upper limit
    let num_parsers = std::cmp::min(16, thread::available_parallelism().map_or(2, std::num::NonZero::get));
    // Store triple indices instead of strings
    let readers = NTriplesParser::new().split_file_for_parallel_parsing(path, num_parsers)?;
    let triples: Vec<[u32; 3]> = readers
        .into_par_iter()
        .flat_map_iter(|reader| {
            reader.filter_map(|q| {
                let q = match q.map_err(|e| Error::new(InvalidData, format!("Error reading N-Triples: {e}"))) {
                    Ok(q) => q,
                    Err(e) => return Some(Err(e)),
                };
                let t = [&dict_string(&q.subject.into()), q.predicate.as_str(), &dict_string(&q.object)];
                intern_triple(&interner, t, policy, &nul_triples)
            })
        })
        .collect::<Result<Vec<[u32; 3]>>>()?;

    let interner = Arc::try_unwrap(interner).expect("interner Arc still has outstanding references");
    Ok((ParsedTerms::new(interner, triples), NtReport { nul_triples: nul_triples.into_inner() }))
}

/// Enumerate the set-bit positions (term indices) of a bitset. Uses
/// `trailing_zeros` per word — far cheaper than iterating every bit and
/// calling `bit_test` (the old `externalize` pattern).
fn collect_set_indices(bitset: &Indices) -> Vec<u32> {
    // Estimate capacity from popcount to avoid Vec grow allocations.
    let popcount: usize = bitset.iter().flat_map(|block| block.iter()).map(|w| w.count_ones() as usize).sum();
    let mut out = Vec::with_capacity(popcount);
    for (block_idx, block) in bitset.iter().enumerate() {
        for (word_idx, &word) in block.iter().enumerate() {
            let base_bit = block_idx * 256 + word_idx * 64;
            let mut w = word;
            while w != 0 {
                let bit_offset = w.trailing_zeros() as usize;
                out.push(u32::try_from(base_bit + bit_offset).expect("term index overflow (>u32::MAX)"));
                w &= w - 1;
            }
        }
    }
    out
}

/// Build the four compressed dictionary sections and the three per-role
/// `index -> HDT id` lookup tables.
///
/// Sections follow the standard HDT MAPPING2 layout:
/// - shared: terms that appear as both subject and object (ids 1..=N_shared for both roles)
/// - unique subjects: subject-only terms (subject ids N_shared+1..=N_shared+N_subj)
/// - unique objects: object-only terms (object ids N_shared+1..=N_shared+N_obj)
/// - predicates: all predicate terms (ids 1..=N_pred)
fn build_dict_and_id_maps(
    terms: &Terms, subjects_bs: &Indices, predicates_bs: &Indices, objects_bs: &Indices, block_size: usize,
) -> (FourSectDict, IdMap, IdMap, IdMap) {
    use log::warn;

    if predicates_bs.is_empty() {
        warn!("no triples found in provided RDF");
    }

    // Compute section membership via bitset ops.
    let mut shared_bs = subjects_bs.clone();
    shared_bs.bit_and(objects_bs);
    let mut unique_subj_bs = subjects_bs.clone();
    unique_subj_bs.bit_andnot(objects_bs);
    let mut unique_obj_bs = objects_bs.clone();
    unique_obj_bs.bit_andnot(subjects_bs);

    // Collect the term indices in each section.
    let mut shared_keys = collect_set_indices(&shared_bs);
    let mut unique_subj_keys = collect_set_indices(&unique_subj_bs);
    let mut pred_keys = collect_set_indices(predicates_bs);
    let mut unique_obj_keys = collect_set_indices(&unique_obj_bs);
    drop(shared_bs);
    drop(unique_subj_bs);
    drop(unique_obj_bs);

    // Sort each section by the resolved string. Each `par_sort_unstable_by`
    // uses the rayon thread pool, so running the four sorts back-to-back lets
    // each one use every core; spawning them all in parallel would just fight
    // over the same workers.
    let cmp = |a: &u32, b: &u32| terms.cmp(*a, *b);
    shared_keys.par_sort_unstable_by(cmp);
    unique_subj_keys.par_sort_unstable_by(cmp);
    pred_keys.par_sort_unstable_by(cmp);
    unique_obj_keys.par_sort_unstable_by(cmp);

    // Allocate ID maps sized by the interner's term count (also the bit
    // length of the role bitsets).
    let map_len = terms.len();
    let mut subj_map: IdMap = vec![0u32; map_len];
    let mut pred_map: IdMap = vec![0u32; map_len];
    let mut obj_map: IdMap = vec![0u32; map_len];

    let n_shared = shared_keys.len();
    let shared_id_ceiling = u32::try_from(n_shared).expect("too many shared terms (>u32::MAX)");
    for (i, &key) in shared_keys.iter().enumerate() {
        let id = (i as u32) + 1; // ids are 1-indexed
        let slot = key as usize;
        subj_map[slot] = id;
        obj_map[slot] = id;
    }
    for (i, &key) in unique_subj_keys.iter().enumerate() {
        subj_map[key as usize] = shared_id_ceiling + (i as u32) + 1;
    }
    for (i, &key) in unique_obj_keys.iter().enumerate() {
        obj_map[key as usize] = shared_id_ceiling + (i as u32) + 1;
    }
    for (i, &key) in pred_keys.iter().enumerate() {
        pred_map[key as usize] = (i as u32) + 1;
    }

    // Compress the four sections concurrently. Each thread pulls its strings
    // straight from the term arena (no intermediate `Vec<&str>` or `BTreeSet`).
    let shared_ref = &shared_keys;
    let unique_subj_ref = &unique_subj_keys;
    let pred_ref = &pred_keys;
    let unique_obj_ref = &unique_obj_keys;
    let (shared, subjects, predicates, objects) = thread::scope(|s| {
        let h_shared = thread::Builder::new()
            .name("shared".into())
            .spawn_scoped(s, || {
                DictSectPFC::compress_iter(shared_ref.iter().map(|&k| terms.get(k)), shared_ref.len(), block_size)
            })
            .unwrap();
        let h_subj = thread::Builder::new()
            .name("unique subjects".into())
            .spawn_scoped(s, || {
                DictSectPFC::compress_iter(
                    unique_subj_ref.iter().map(|&k| terms.get(k)),
                    unique_subj_ref.len(),
                    block_size,
                )
            })
            .unwrap();
        let h_pred = thread::Builder::new()
            .name("predicates".into())
            .spawn_scoped(s, || {
                DictSectPFC::compress_iter(pred_ref.iter().map(|&k| terms.get(k)), pred_ref.len(), block_size)
            })
            .unwrap();
        let h_obj = thread::Builder::new()
            .name("unique objects".into())
            .spawn_scoped(s, || {
                DictSectPFC::compress_iter(
                    unique_obj_ref.iter().map(|&k| terms.get(k)),
                    unique_obj_ref.len(),
                    block_size,
                )
            })
            .unwrap();
        (h_shared.join().unwrap(), h_subj.join().unwrap(), h_pred.join().unwrap(), h_obj.join().unwrap())
    });

    (FourSectDict { shared, subjects, predicates, objects }, subj_map, pred_map, obj_map)
}

#[cfg(test)]
pub mod tests {
    use super::super::StringTriple;
    use super::super::tests::snikmeta_check;
    use super::{Hdt, NtOptions, NulPolicy};
    use crate::hdt::tests::snikmeta;
    use crate::tests::init;
    use color_eyre::Result;
    use fs_err::File;
    use pretty_assertions::assert_eq;
    use std::io::{Cursor, Write};
    use std::path::Path;
    use std::sync::Arc;

    /// Regression test for #131. Escapes are N-Triples syntax, not part of a term's value,
    /// so the dictionary must hold the decoded value and each serialization must escape it exactly once.
    #[test]
    fn read_nt_escapes() -> Result<()> {
        init();
        // tests/resources/escapes.nt in dictionary string format
        let want: Vec<StringTriple> = [
            ["_:b0", "urn:x:bnode", "\"x\""],
            ["urn:x:s", "urn:x:backslash", "\"back\\slash\"^^<urn:x:dt>"],
            ["urn:x:s", "urn:x:cr", "\"carriage\rreturn\""],
            ["urn:x:s", "urn:x:iri", "urn:x:oé"],
            ["urn:x:s", "urn:x:newline", "\"line\nbreak\"@en"],
            ["urn:x:s", "urn:x:nul1", "\"firsta\""],
            ["urn:x:s", "urn:x:nul4", "\"aa\""],
            ["urn:x:s", "urn:x:nul4", "\"ab\""],
            ["urn:x:s", "urn:x:plain", "\"nothing to escape\""],
            ["urn:x:s", "urn:x:quote", "\"say \"hi\"\""],
            ["urn:x:s", "urn:x:raw", "\"café 😀\""],
            ["urn:x:s", "urn:x:tab", "\"tab\there\""],
            ["urn:x:s", "urn:x:unicode", "\"café 😀\""],
        ]
        .map(|t| t.map(Arc::from))
        .into();

        let drop = NtOptions::default().nul_policy(NulPolicy::Drop);
        let (from_nt, report) = Hdt::read_nt_with("tests/resources/escapes.nt", &drop)?;
        assert_eq!(from_nt.triples_all().collect::<Vec<_>>(), want, "dictionary must hold decoded values");
        assert_eq!(report.nul_triples, 6);

        // the same graph given as decoded strings must build the same HDT
        let mut wantmore = want.clone();
        wantmore.push(["urn:x:s", "urn:x:nul4", "\"a\u{0000}\"bc"].map(Arc::from));
        let (from_triples, _) = Hdt::from_triples_with(wantmore, "urn:x:escapes", &drop)?;
        assert_eq!(from_triples.triple_ids_with_pattern(Some("urn:x:s"), None, Some("\"aa\"")).count(), 1);
        assert_eq!(from_triples.triples_all().collect::<Vec<_>>(), want);
        assert_eq!(from_triples.triples.bitmap_y.dict, from_nt.triples.bitmap_y.dict);

        // the reported symptom: out to N-Triples and back in must not change a value
        fs_err::create_dir_all("tests/resources/generated")?;
        let path = Path::new("tests/resources/generated/escapes.nt");
        let mut writer = std::io::BufWriter::new(File::create(path)?);
        from_nt.write_nt(&mut writer)?;
        writer.flush()?;
        assert_eq!(Hdt::read_nt(path)?.triples_all().collect::<Vec<_>>(), want, "NT must escape once");

        let mut buf = Vec::<u8>::new();
        from_nt.write(&mut buf)?;
        let again = Hdt::read(Cursor::new(buf))?.triples_all().collect::<Vec<_>>();
        assert_eq!(again, want, "HDT must preserve values");
        Ok(())
    }

    fn write_generated(name: &str, content: &[u8]) -> Result<std::path::PathBuf> {
        fs_err::create_dir_all("tests/resources/generated")?;
        let path = Path::new("tests/resources/generated").join(name);
        fs_err::write(&path, content)?;
        Ok(path)
    }

    fn assert_nul_rejected<T: std::fmt::Debug>(r: std::io::Result<T>) -> String {
        let e = r.expect_err("nul char must be rejected by default");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("nul char"), "{e}");
        e.to_string()
    }

    /// HDT cannot store U+0000, so by default it is an error however it is spelled.
    #[test]
    fn nul_rejected_by_default() -> Result<()> {
        init();
        assert_nul_rejected(Hdt::read_nt("tests/resources/escapes.nt"));
        for (name, o) in
            [("u4", &br#""a\u0000b""#[..]), ("u8", &br#""a\U00000000b""#[..]), ("raw", &b"\"a\0b\""[..])]
        {
            let nt = [&b"<urn:x:s> <urn:x:p> \"ok\" .\n<urn:x:s> <urn:x:p> "[..], o, b" .\n"].concat();
            assert_nul_rejected(Hdt::read_nt(write_generated(&format!("nul_reject_{name}.nt"), &nt)?));
        }
        // subject, predicate and object each go to a different dictionary section
        for (i, bad) in
            [["urn:x:\0", "urn:x:p", "\"o\""], ["urn:x:s", "urn:x:\0", "\"o\""], ["urn:x:s", "urn:x:p", "\"\0\""]]
                .into_iter()
                .enumerate()
        {
            let e = assert_nul_rejected(Hdt::from_triples([["urn:x:s", "urn:x:p", "\"o\""], bad], "urn:x:nul"));
            assert!(e.starts_with("triple 1: "), "position {i}: {e}");
        }
        // the error shows the whole triple, with the nul visible, to find it in the input
        let e = assert_nul_rejected(Hdt::from_triples([["urn:x:s", "urn:x:p", "\"a\0b\"@en"]], "urn:x:nul"));
        assert_eq!(
            e,
            r#"triple 0: <urn:x:s> <urn:x:p> "a\u0000b"@en has a nul char (U+0000) in the object, which HDT cannot store"#
        );
        Ok(())
    }

    /// A nul char outside a literal value is invalid RDF, so even the lossy policies reject it
    /// instead of making up a different identifier.
    #[test]
    fn nul_outside_literal_value_always_rejected() {
        init();
        for policy in [NulPolicy::Reject, NulPolicy::Drop, NulPolicy::Truncate, NulPolicy::Strip] {
            let opts = NtOptions::default().nul_policy(policy);
            for bad in [
                ["urn:x:s\0", "urn:x:p", "\"o\""],
                ["_:b\0", "urn:x:p", "\"o\""],
                ["urn:x:s", "urn:x:p\0", "\"o\""],
                ["urn:x:s", "urn:x:p", "urn:x:o\0"],
                ["urn:x:s", "urn:x:p", "\"o\"@en\0"],
                ["urn:x:s", "urn:x:p", "\"o\"^^<urn:x:dt\0>"],
            ] {
                let e = Hdt::from_triples_with([["urn:x:s", "urn:x:p", "\"ok\""], bad], "urn:x:nul", &opts)
                    .expect_err(&format!("{policy:?} {bad:?}"));
                assert!(e.to_string().starts_with("triple 1: "), "{policy:?} {bad:?}: {e}");
                assert!(e.to_string().contains("not valid"), "{policy:?} {bad:?}: {e}");
            }
        }
    }

    /// Each lossy policy must rewrite or drop terms before interning, so that terms it makes
    /// equal are deduplicated and the dictionary stays sorted: `"a\0b"` sorts before `"aa"`
    /// but `"ab"` after it, and `"a\0a"` becomes a duplicate of `"aa"` under Strip.
    #[test]
    fn nul_lossy_policies() -> Result<()> {
        init();
        let nt =
            b"<urn:x:s> <urn:x:p> \"a\\u0000b\" .\n<urn:x:s> <urn:x:p> \"aa\" .\n<urn:x:s> <urn:x:p> \"ab\" .\n\
<urn:x:s> <urn:x:p> \"ab\\U00000000\"@en .\n<urn:x:s> <urn:x:p> \"a\0a\" .\n<urn:x:s> <urn:x:p> \"ab\"@en .\n";
        let path = write_generated("nul_lossy.nt", nt)?;
        let input = ["\"a\0b\"", r#""aa""#, r#""ab""#, "\"ab\0\"@en", "\"a\0a\"", "\"ab\"@en"];
        let triples: Vec<[&str; 3]> = input.iter().map(|&o| ["urn:x:s", "urn:x:p", o]).collect();

        for (policy, objects) in [
            (NulPolicy::Drop, &[r#""aa""#, r#""ab""#, r#""ab"@en"#][..]),
            (NulPolicy::Truncate, &[r#""a""#, r#""aa""#, r#""ab""#, r#""ab"@en"#][..]),
            (NulPolicy::Strip, &[r#""aa""#, r#""ab""#, r#""ab"@en"#][..]),
        ] {
            let opts = NtOptions::default().nul_policy(policy);
            let want_nt: Vec<StringTriple> =
                objects.iter().map(|&o| ["urn:x:s", "urn:x:p", o].map(Arc::from)).collect();

            let (from_nt, report) = Hdt::read_nt_with(&path, &opts)?;
            assert_eq!(report.nul_triples, 3, "{policy:?}");
            let (from_mem, report) = Hdt::from_triples_with(triples.clone(), "urn:x:nul", &opts)?;
            assert_eq!(report.nul_triples, 3, "{policy:?}");

            let mut buf = Vec::<u8>::new();
            from_mem.write(&mut buf)?;
            let reread = Hdt::read(Cursor::new(buf))?;

            for (hdt, want, what) in [
                (&from_nt, &want_nt, "read_nt_with"),
                (&from_mem, &want_nt, "from_triples_with"),
                (&reread, &want_nt, "reread"),
            ] {
                assert_eq!(hdt.triples_all().collect::<Vec<_>>(), *want, "{policy:?} {what}");
                for [s, p, o] in want {
                    let found: Vec<_> = hdt.triples_with_pattern(Some(s), None, Some(o)).collect();
                    assert_eq!(found, [[s.clone(), p.clone(), o.clone()]], "{policy:?} {what} lookup {o}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn read_nt() -> Result<()> {
        init();
        let path = Path::new("tests/resources/snikmeta.nt");
        if !path.exists() {
            log::info!("Creating test resource snikmeta.nt.");
            let mut writer = std::io::BufWriter::new(File::create(path)?);
            snikmeta()?.write_nt(&mut writer)?;
        }
        let invalid = path.join("doesnotexist");
        assert!(Hdt::read_nt(invalid).is_err(), "invalid N-Triples path should result in error");

        let snikmeta_nt = Hdt::read_nt(path)?;
        let snikmeta = snikmeta()?;
        let hdt_triples: Vec<StringTriple> = snikmeta.triples_all().collect();
        let nt_triples: Vec<StringTriple> = snikmeta_nt.triples_all().collect();

        assert_eq!(nt_triples, hdt_triples);
        assert_eq!(snikmeta.triples.bitmap_y.dict, snikmeta_nt.triples.bitmap_y.dict);
        snikmeta_check(&snikmeta_nt)?;
        let path = Path::new("tests/resources/empty.nt");
        let hdt_empty = Hdt::read_nt(path)?;
        let mut buf = Vec::<u8>::new();
        hdt_empty.write(&mut buf)?;
        Hdt::read(Cursor::new(buf))?;
        Ok(())
    }

    #[test]
    fn from_triples() -> Result<()> {
        init();
        let snikmeta = snikmeta()?;
        let triples: Vec<StringTriple> = snikmeta.triples_all().collect();
        let from_triples = Hdt::from_triples(triples, "http://www.snik.eu/ontology/meta")?;

        let hdt_triples: Vec<StringTriple> = snikmeta.triples_all().collect();
        let mem_triples: Vec<StringTriple> = from_triples.triples_all().collect();
        assert_eq!(mem_triples, hdt_triples);
        assert_eq!(snikmeta.triples.bitmap_y.dict, from_triples.triples.bitmap_y.dict);
        snikmeta_check(&from_triples)?;
        let mut buf = Vec::<u8>::new();
        from_triples.write(&mut buf)?;
        snikmeta_check(&Hdt::read(Cursor::new(buf))?)?;
        let hdt_empty = Hdt::from_triples(std::iter::empty::<[&str; 3]>(), "http://example.org/empty")?;
        let mut buf = Vec::<u8>::new();
        hdt_empty.write(&mut buf)?;
        Hdt::read(Cursor::new(buf))?;
        Ok(())
    }
}
