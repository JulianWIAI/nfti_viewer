/**
 * neuroData.ts — Biological dictionary for the Contextual Reference Panel
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Provides expert-level neuroscientific descriptions for all 32 structures in
 * the FreeSurfer SynthSeg segmentation (labels 2–60; background label 0 excluded),
 * plus MEG electrophysiology and fMRI haemodynamic methodology.
 *
 * HEX COLOURS
 * ────────────
 * Derived pixel-for-pixel from the SYNTHSEG_COLORS table in
 * src/lib/vtk/segmentationOverlay.ts so that every colour swatch in the
 * Reference Drawer matches the rendered 3-D overlay exactly.
 *
 * FUNCTIONAL CONTENT SOURCES
 * ───────────────────────────
 * Kandel, Principles of Neural Science 6e; Purves, Neuroscience 6e;
 * Friston, The free-energy principle (Nat Rev Neurosci 2010);
 * MNE-Python developer documentation.
 */

// ── Type definitions ───────────────────────────────────────────────────────────

/**
 * A single anatomical region within a group.
 * Labels that span both hemispheres list both IDs (left label first).
 */
export interface NeuroRegion {
  /** Unique camelCase key used for context navigation and accordion state. */
  id: string;
  /** Display name shown in the accordion row. */
  name: string;
  /**
   * FreeSurfer SynthSeg label IDs for this structure.
   * Bilateral structures have two entries: [leftLabel, rightLabel].
   * Used to build the label→region lookup map in ReferencePanelContext.
   */
  fsLabels: number[];
  /**
   * Hex colour derived from the left-hemisphere entry in SYNTHSEG_COLORS.
   * Pixel-identical to the colour rendered by segmentationOverlay.ts.
   */
  color: string;
  /** One-line functional tag for the accordion header and search matching. */
  function: string;
  /**
   * Extended neuroscientific detail paragraph shown when the region row is
   * expanded. Includes circuitry, connectivity, and clinical relevance.
   */
  detail: string;
}

/** A thematic grouping of related anatomical regions. */
export interface NeuroGroup {
  /** Unique kebab-case key. */
  id: string;
  /** Human-readable header label. */
  label: string;
  /** Emoji icon prepended to the group header. */
  icon: string;
  /** Short sentence summarising the group's shared computational role. */
  description: string;
  /** Ordered list of regions belonging to this group. */
  regions: NeuroRegion[];
}

/** One frequency band in the MEG electrophysiology section. */
export interface MegBand {
  id: string;
  label: string;
  /** Formatted frequency range string (e.g. "4–8 Hz"). */
  freqRange: string;
  freqMinHz: number;
  /** null for open-ended ranges (e.g. gamma "≥ 30 Hz"). */
  freqMaxHz: number | null;
  /** One-line functional tag. */
  function: string;
  /** Extended description of oscillatory mechanisms and cognitive roles. */
  detail: string;
}

/** An individual ERP waveform (e.g. MMN, P300). */
export interface ErpWave {
  id: string;
  label: string;
  /** Canonical latency window (e.g. "100–200 ms post-deviant"). */
  latency: string;
  /** Typical scalp distribution and amplitude. */
  distribution: string;
  /** Cognitive/computational interpretation. */
  interpretation: string;
  /** Underlying mathematical / computational-neuroscience framing. */
  mathematicalBasis: string;
}

/** The ERP accordion entry containing multiple waveform definitions. */
export interface ErpEntry {
  id: string;
  label: string;
  description: string;
  waves: ErpWave[];
}

/** Full MEG electrophysiology section of the dictionary. */
export interface MegElectrophysiology {
  bands: MegBand[];
  erp: ErpEntry;
}

/** fMRI haemodynamics methodology entry. */
export interface FmriHemodynamics {
  id: string;
  label: string;
  /** One-line tag. */
  function: string;
  /** Full HRF pipeline description and design implications. */
  detail: string;
}

/** Root dictionary structure exported for consumption by the Reference Drawer. */
export interface NeuroDictionary {
  anatomy: NeuroGroup[];
  megElectrophysiology: MegElectrophysiology;
  fmriHemodynamics: FmriHemodynamics;
}

// ── Anatomy ────────────────────────────────────────────────────────────────────

const anatomyGroups: NeuroGroup[] = [

  // ── 1. Cortical Network ─────────────────────────────────────────────────────
  // SynthSeg uses a single label per hemisphere for the entire cortical ribbon:
  //   label 3 = left cerebral cortex, label 42 = right cerebral cortex.
  //   label 2 = left cerebral white matter, label 41 = right cerebral white matter.
  // The four lobe entries are conceptual teaching subdivisions of one SynthSeg
  // label pair (3/42); clicking any cortical voxel in the 3-D viewer navigates
  // here.  White matter is a distinct SynthSeg class with its own entry below.
  {
    id: 'cortical-network',
    label: 'Cortical Network',
    icon: '🧠',
    description: 'The cerebral cortex and its underlying white matter highways — 2–4 mm of layered neocortex performing high-order cognition, connected by myelinated axon tracts.',
    regions: [
      {
        id: 'frontal-lobe',
        name: 'Frontal Lobe',
        fsLabels: [3, 42],  // shares whole-cortex label in SynthSeg
        color: '#CD3E4E',   // rgb(205,62,78) — left cerebral cortex in overlay
        function: 'Rule memory, motor planning, error prediction',
        detail:
          'Prefrontal cortex (PFC) maintains abstract conditional rules in working memory ("if A, then B") via persistent firing in layer-III pyramidal cells. Anterior cingulate cortex (ACC) — specifically the dorsal ACC — generates forward-model prediction-error signals when an expected outcome is violated; its firing correlates with the MMN/P300 in MEG. Supplementary motor area (SMA) drives internally-timed motor sequence initiation roughly 500 ms before voluntary movement (the Bereitschaftspotential). Fronto-parietal beta synchrony (13–30 Hz) transmits top-down predictions in predictive-coding frameworks.',
      },
      {
        id: 'parietal-lobe',
        name: 'Parietal Lobe',
        fsLabels: [3, 42],
        color: '#CD3E4E',
        function: 'Spatial representation, quantity coding, attention',
        detail:
          'Intraparietal sulcus (IPS) houses tuned neurons for non-symbolic magnitude and visuospatial computation required for geometric reasoning. Angular gyrus maps abstract number symbols to quantity representations — lesions cause acalculia and Gerstmann syndrome. Posterior parietal cortex (PPC) constructs egocentric and allocentric spatial reference frames; damage causes hemispatial neglect and optic ataxia. P3b (300–500 ms ERP) is maximal over parietal electrodes and indexes context updating of an internal world-model.',
      },
      {
        id: 'temporal-lobe',
        name: 'Temporal Lobe',
        fsLabels: [3, 42],
        color: '#CD3E4E',
        function: 'Auditory sequence processing, semantic retrieval',
        detail:
          'Superior temporal gyrus (STG) and planum temporale perform spectro-temporal fine-structure analysis of speech and music: frequency tuning, onset detection, and phonological parsing. Wernicke\'s area (posterior STG) integrates phonological and semantic streams. Middle temporal gyrus (MTG) supports semantic retrieval and conceptual knowledge. In MEG, Mismatch Negativity (MMN; peak ~150 ms) originates in bilateral STG and indexes automatic detection of acoustic deviants against a learned auditory template. Theta-gamma coupling between STG and PFC coordinates sentence-level syntactic parsing.',
      },
      {
        id: 'occipital-lobe',
        name: 'Occipital Lobe',
        fsLabels: [3, 42],
        color: '#CD3E4E',
        function: 'Visual feature extraction, spatial frequency, motion',
        detail:
          'Primary visual cortex (V1, striate cortex) retinotopically encodes orientation columns, ocular-dominance columns, spatial frequency, and color via simple/complex cell hierarchies. V2 and V3 build intermediate features. V4 (ventral stream) handles color constancy, object permanence, and texture — lesions cause cerebral achromatopsia. V5/MT (dorsal stream) encodes motion vectors and optical flow — lesions cause akinetopsia. Occipital alpha power (8–10 Hz) inversely predicts visual cortex excitability: alpha suppression = cortical activation.',
      },
      {
        id: 'cerebral-white-matter',
        name: 'Cerebral White Matter',
        fsLabels: [2, 41],  // label 2 = left, 41 = right in SynthSeg
        color: '#F5F5F5',   // rgb(245,245,245) — near-white, matches overlay exactly
        function: 'Cortico-cortical and cortico-subcortical myelinated axon highways',
        detail:
          'White matter occupies roughly half the cerebral volume and is subdivided into three functional fibre classes. (1) Association fibres link cortical areas within the same hemisphere: the arcuate fasciculus (AF) bridges Broca\'s area (left inferior frontal gyrus) and Wernicke\'s area (posterior STG) — damage causes conduction aphasia; the superior longitudinal fasciculus (SLF) runs between frontal and parietal-temporal cortex, supporting visuospatial attention (SLF-II/III) and language (SLF-I); the uncinate fasciculus connects orbitofrontal cortex to the temporal pole, carrying social-emotional–memory integration signals; the inferior fronto-occipital fasciculus (IFOF) carries semantic information from posterior cortex to frontal pole; the cingulum bundles hippocampal, parahippocampal, and cingulate fibres into a C-shaped limbic highway. (2) Commissural fibres cross the midline — the corpus callosum (~250 million axons, genu → body → splenium) synchronises bilateral cortical hemispheres and allows interhemispheric transfer of motor, somatosensory, visual, and language information; agenesis of the corpus callosum disrupts interhemispheric transfer time (disconnection syndrome). (3) Projection fibres link cortex to subcortical structures and spinal cord — the corticospinal tract (CST) descends through the posterior limb of the internal capsule carrying voluntary motor commands; the corticobulbar tract controls cranial-nerve motor nuclei (facial and speech muscles); thalamocortical radiations (optic radiations, acoustic radiations, etc.) convey sensory signals upward to primary sensory cortices. Myelination by oligodendrocytes increases conduction velocity up to 150 m/s and lowers metabolic cost per impulse. Diffusion Tensor Imaging (DTI) resolves tract architecture non-invasively by quantifying the directional anisotropy of water diffusion: fractional anisotropy (FA, 0–1; higher = more organised tract), mean diffusivity (MD), and tract streamlines visualised via tractography. Reduced FA in the AF predicts reading disability; reduced FA in the cingulum is a transdiagnostic biomarker for depression and anxiety. White Matter Hyperintensities (WMH) on T2-FLAIR MRI — bright oval foci adjacent to ventricles or in deep WM — mark small-vessel ischaemia, demyelination (MS), or diffuse axonal injury; cumulative WMH burden correlates with slowed processing speed, gait disturbance, and vascular dementia risk.',
      },
    ],
  },

  // ── 2. Basal Ganglia ────────────────────────────────────────────────────────
  // A set of subcortical nuclei that form cortico-striato-thalamo-cortical
  // (CSTC) loops governing action selection, sequence learning, and timing.
  {
    id: 'basal-ganglia',
    label: 'Basal Ganglia',
    icon: '⚙️',
    description: 'Subcortical nuclei forming CSTC loops that select, sequence, and time motor and cognitive actions via dopamine-gated competition.',
    regions: [
      {
        id: 'caudate',
        name: 'Caudate Nucleus',
        fsLabels: [11, 50],
        color: '#7ABADC',   // rgb(122,186,220)
        function: 'Sequence syntax — associative cortico-striatal loop',
        detail:
          'Head of the caudate nucleus receives dense glutamatergic projections from dorsolateral PFC forming the associative CSTC loop. Learns action–outcome contingencies via dopaminergic prediction-error signals (δ = r − V) from substantia nigra pars compacta. Stores the abstract "syntax" of learned sequences — which action follows which — independently of the specific motor parameters (stored in putamen). Caudate volume predicts second-language syntactic proficiency. Caudate body/tail connect to visual cortex and participate in visual habit learning.',
      },
      {
        id: 'putamen',
        name: 'Putamen',
        fsLabels: [12, 51],
        color: '#EC0DB0',   // rgb(236,13,176)
        function: 'Sub-second interval timing — sensorimotor loop',
        detail:
          'Largest nucleus of the dorsal striatum; receives somatosensory and primary motor cortex projections forming the sensorimotor CSTC loop. Critically involved in sub-second interval timing (beat induction, 200 ms–2 s range), procedural motor skill acquisition (sequence speed and automaticity), and speech articulation timing. Parkinson\'s disease cardinal motor symptoms arise from dopaminergic depletion of putamen: bradykinesia, rigidity, and tremor. Deep brain stimulation (DBS) of subthalamic nucleus effectively decouples the pathologically synchronized beta oscillations (13–30 Hz) in the putamen–GPi–thalamus loop.',
      },
      {
        id: 'pallidum',
        name: 'Globus Pallidus',
        fsLabels: [13, 52],
        color: '#0C30FF',   // rgb(12,48,255)
        function: 'Output gating — inhibitory valve on thalamo-cortical loops',
        detail:
          'Constitutes the primary output nucleus of the basal ganglia. Globus pallidus internus (GPi) tonically inhibits thalamo-cortical circuits via high-frequency (~80 Hz) GABAergic firing. Action selection occurs when striatal direct-pathway D1 neurons disinhibit GPi ("release the brake"), allowing thalamo-cortical excitation of the winning motor program. GPe (external) participates in the indirect (suppressing) and hyperdirect (STN-mediated fast suppression) pathways. Surgical GPi ablation (pallidotomy) reduces Parkinson\'s tremor and dyskinesias. Pathological GPi over-inhibition underlies bradykinesia.',
      },
      {
        id: 'accumbens',
        name: 'Nucleus Accumbens',
        fsLabels: [26, 58],
        color: '#FFA500',   // rgb(255,165,0)
        function: 'Prediction error / reward — limbic–motor interface',
        detail:
          'Ventral striatum; interface between limbic emotion processing (amygdala, hippocampus) and motor output systems. Receives mesolimbic dopamine from VTA encoding reward prediction error (RPE = r − V̂, where V̂ is the learned expected value). Shell region processes novelty and aversion; core region mediates Pavlovian-to-instrumental transfer (how conditioned cues trigger habitual behaviour). Central to addiction neurobiology: repeated drug exposure compulsively potentiates AMPA/NMDA synaptic weights in the accumbens core, creating pathologically inflated prediction-error signals that override prefrontal control.',
      },
    ],
  },

  // ── 3. Limbic System ────────────────────────────────────────────────────────
  // Phylogenetically older cortical and subcortical structures mediating
  // memory consolidation, emotional salience, and autonomic drive.
  {
    id: 'limbic-system',
    label: 'Limbic System',
    icon: '💛',
    description: 'Evolutionarily conserved structures linking sensory experience to emotional valence, long-term memory consolidation, and autonomic drive.',
    regions: [
      {
        id: 'hippocampus',
        name: 'Hippocampus',
        fsLabels: [17, 53],
        color: '#DCD814',   // rgb(220,216,20)
        function: 'Episodic/spatial memory binding — pattern completion & separation',
        detail:
          'Performs relational memory binding: links "what", "where", and "when" features into a unified episodic memory trace. CA3 performs pattern completion (retrieve full memory from partial cue) via recurrent collaterals. Dentate gyrus performs pattern separation (distinguish similar events) via sparse coding. CA1 computes match–mismatch between CA3 completion and current entorhinal input, generating prediction-error signals. Place cells encode allocentric location; grid cells (entorhinal) provide a metric scaffold for cognitive maps. Hippocampal–PFC theta coherence (4–8 Hz) gates memory consolidation during waking replay and NREM slow-wave sleep. MTL lesions (Patient H.M.) cause anterograde amnesia with preserved procedural learning.',
      },
      {
        id: 'amygdala',
        name: 'Amygdala',
        fsLabels: [18, 54],
        color: '#67FFFF',   // rgb(103,255,255)
        function: 'Threat detection, emotional salience, fear conditioning',
        detail:
          'Basolateral amygdala (BLA) receives polymodal sensory input from cortex and thalamus (fast subcortical "low road" bypasses neocortex for rapid threat response). BLA assigns emotional salience to stimuli via Hebbian plasticity; projects to hippocampus to modulate consolidation of emotionally significant episodic memories — explaining heightened memory for fearful events. Central nucleus (CeA) drives autonomic fear responses (heart rate, HPA axis) via hypothalamus and brain stem. BLA shows gamma oscillations (>60 Hz) during threat processing. Amygdala hyperactivity is the primary biomarker in PTSD and anxiety disorders; ketamine and MDMA reduce CeA hyperactivity via distinct mechanisms.',
      },
      {
        id: 'ventral-dc',
        name: 'Ventral Diencephalon',
        fsLabels: [28, 60],
        color: '#A52A2A',   // rgb(165,42,42)
        function: 'Autonomic regulation, dopamine / norepinephrine sources',
        detail:
          'Encompasses the ventral diencephalon including substantia nigra pars compacta (SNc — mesostrial dopamine, motor control), ventral tegmental area (VTA — mesolimbic/mesocortical dopamine, reward/cognition), subthalamic nucleus (STN — motor control via hyperdirect pathway), hypothalamus (homeostatic drives, HPA axis), and red nucleus (rubrospinal motor control). SNc degeneration is the pathological hallmark of Parkinson\'s disease. STN is the primary target for DBS. VTA firing encodes RPE and drives the nucleus accumbens. Locus coeruleus (adjacent, norepinephrine) modulates cortical signal-to-noise ratio and arousal.',
      },
    ],
  },

  // ── 4. Relay Centers ────────────────────────────────────────────────────────
  // Structures that preprocess, route, and time-stamp neural signals before
  // delivery to cortex.
  {
    id: 'relay-centers',
    label: 'Relay & Timing Centers',
    icon: '📡',
    description: 'Thalamus, cerebellum, and brain stem relay and pre-process information — providing the sensory gateway, sub-millisecond timing engine, and arousal scaffold for cortex.',
    regions: [
      {
        id: 'thalamus',
        name: 'Thalamus',
        fsLabels: [10, 49],
        color: '#00760E',   // rgb(0,118,14)
        function: 'Sensory routing hub — modality-specific relay nuclei',
        detail:
          'Universal sensory gateway to neocortex. Modality-specific relay nuclei: medial geniculate nucleus (MGN) → primary auditory cortex (A1, temporal); lateral geniculate nucleus (LGN) → primary visual cortex (V1, occipital); ventral posterolateral nucleus (VPL) → somatosensory cortex (S1, parietal); ventral anterior / ventrolateral nuclei → motor cortex (frontal). Non-specific nuclei: pulvinar modulates attention across association cortex; mediodorsal nucleus (MD) → PFC for working memory. Thalamo-cortical relay cells fire in two modes: tonic (awake, single spikes — faithful relay) and burst mode (hyperpolarized via T-type Ca²⁺ channels — spindle/alpha generation during sleep). Sleep spindles (12–15 Hz) and alpha rhythm originate from thalamo-cortical resonance.',
      },
      {
        id: 'cerebellar-cortex',
        name: 'Cerebellar Cortex',
        fsLabels: [8, 47],
        color: '#E79422',   // rgb(231,148,34)
        function: 'Sub-millisecond sequence timing — supervised motor learning',
        detail:
          'Three-layer cortex with a single output cell type: Purkinje cells (sole GABAergic output). Receives two fundamentally different inputs: (1) mossy fiber → granule cell → parallel fiber (context encoding, ~100 billion granule cells) and (2) climbing fiber from inferior olive (error signal, one-to-one with each Purkinje). Cerebellar learning implements gradient descent via LTD at parallel-fiber synapses when climbing-fiber error signals coincide — the biological substrate of supervised learning. Purkinje cells achieve sub-millisecond timing precision (10–100 ms resolution) critical for speech articulation, rhythmic motor programs, and anticipatory smooth-pursuit eye movements. Cerebellar ataxia disrupts timing and coordination while leaving force generation intact.',
      },
      {
        id: 'cerebellar-wm',
        name: 'Cerebellar White Matter',
        fsLabels: [7, 46],
        color: '#DCF8A4',   // rgb(220,248,164)
        function: 'Deep cerebellar nuclei — output pathway to thalamus',
        detail:
          'Contains the axons of Purkinje cells projecting to the deep cerebellar nuclei (DCN), which are the primary output stage of the cerebellum. Three DCN: (1) dentate nucleus (largest, projects via superior cerebellar peduncle → red nucleus / VL thalamus → motor cortex), (2) interpositus nucleus (emboliform + globose; reflex conditioning, somatosensory feedback), (3) fastigial nucleus (vestibular / postural control, projects to vestibulocerebellum and spinal cord). DCN neurons fire tonically at ~50–100 Hz and are rate-modulated by Purkinje cell inhibition — fast inhibition (Purkinje) of fast excitation (DCN) enables precise timing via release-from-inhibition.',
      },
      {
        id: 'brain-stem',
        name: 'Brain Stem',
        fsLabels: [16],
        color: '#779FB0',   // rgb(119,159,176)
        function: 'Cochlear nucleus, auditory colliculus, arousal nuclei',
        detail:
          'First-stage auditory processing: cochlear nuclei (CN) receive VIII-nerve (spiral ganglion) input and compute onset detection, frequency tuning, and temporal fine structure. Dorsal CN → contralateral inferior colliculus (IC); ventral CN → bilateral superior olive (binaural ITD/ILD computation for sound localization). Inferior colliculus integrates all ascending auditory streams and projects to medial geniculate nucleus (MGN). Arousal nuclei: locus coeruleus (LC; norepinephrine) modulates cortical gain/signal-to-noise; raphe nuclei (serotonin) regulate mood, sleep architecture, and temporal integration of sensory streams. Auditory brainstem response (ABR) waves I–V map onto CN, olivary, and collicular generators, allowing non-invasive assay of peripheral hearing function.',
      },
    ],
  },

  // ── 5. Ventricular System ───────────────────────────────────────────────────
  // CSF-filled spaces — passive compartments important for intracranial
  // pressure regulation and metabolite clearance; enlarged/displaced
  // compartments are key radiological diagnostic markers.
  {
    id: 'ventricular-system',
    label: 'Ventricular System',
    icon: '💧',
    description: 'CSF-filled cavities providing buoyancy, metabolite clearance via the glymphatic system, and ICP buffering; morphological changes are key diagnostic markers.',
    regions: [
      {
        id: 'lateral-ventricle',
        name: 'Lateral Ventricles',
        fsLabels: [4, 43],
        color: '#1E76A1',   // rgb(30,118,161) — left label from overlay
        function: 'Primary CSF production via choroid plexus',
        detail:
          'C-shaped cavities that trace the head, body, and tail of the caudate nucleus. Choroid plexus (vascular tufts in the temporal and body segments) secretes ~500 mL CSF/day at 20 mL/hr. CSF drains through the foramina of Monro into the 3rd ventricle. Enlargement indicates cerebral atrophy (sulcal widening + enlarged ventricles = normal-pressure hydrocephalus or dementia). Ventricular volume correlates negatively with cortical thickness — a proxy for brain age. Periventricular white matter lesions (PVWML) adjacent to lateral ventricles are hyperintense on FLAIR and mark small-vessel ischaemia or demyelination.',
      },
      {
        id: 'inf-lat-ventricle',
        name: 'Inferior Lateral Ventricles',
        fsLabels: [5, 44],
        color: '#653C80',   // rgb(101,60,128)
        function: 'Temporal horn CSF drainage',
        detail:
          'Inferior (temporal) horn of the lateral ventricles; curves anteroinferiorly to lie within the temporal lobe adjacent to the hippocampal formation. Enlargement of the temporal horn is an early MRI biomarker of hippocampal atrophy in Alzheimer\'s disease — the hippocampus compresses the inferior horn; as it shrinks, the horn expands. This measurement (temporal horn width > 3 mm on 1.5 T MRI) is part of the Scheltens scoring system for medial temporal lobe atrophy.',
      },
      {
        id: 'third-ventricle',
        name: '3rd Ventricle',
        fsLabels: [14],
        color: '#CCB68E',   // rgb(204,182,142)
        function: 'Diencephalic CSF space — thalamic interadhesion',
        detail:
          'Midline slit-like cavity flanked bilaterally by the thalami; separated from 4th ventricle by the cerebral aqueduct of Sylvius (the narrowest CSF passage, 1–2 mm diameter). The massa intermedia (interthalamic adhesion) traverses the 3rd ventricle in ~70% of humans. CSF obstruction at the aqueduct causes obstructive hydrocephalus with dilated lateral/3rd but normal 4th ventricles. Neuroendoscopic third ventriculostomy (ETV) bypasses aqueductal stenosis by creating a perforation in the floor of the 3rd ventricle.',
      },
      {
        id: 'fourth-ventricle',
        name: '4th Ventricle',
        fsLabels: [15],
        color: '#2ACCA4',   // rgb(42,204,164)
        function: 'Posterior fossa CSF egress via foramina',
        detail:
          'Rhomboid cavity overlying the pons and medulla (floor) and beneath the cerebellum (roof). CSF exits into the subarachnoid space via the paired foramina of Luschka (lateral) and the midline foramen of Magendie (caudal). Dandy–Walker malformation (aplastic vermis, cystic dilation of 4th ventricle) and Chiari malformation (inferior cerebellar tonsillar herniation through the foramen magnum compressing the 4th ventricle) are the main congenital pathologies identified in this region.',
      },
      {
        id: 'csf',
        name: 'CSF Space',
        fsLabels: [24],
        color: '#3C3C3C',   // rgb(60,60,60)
        function: 'Buoyancy, glymphatic metabolite clearance, immune surveillance',
        detail:
          'Total CSF volume ~150 mL (35 mL ventricular, 115 mL subarachnoid). Buoyancy effect reduces effective brain weight from ~1 400 g (in air) to ~50 g — preventing the brain crushing its own blood supply under gravity. Glymphatic system (sleep-dependent aquaporin-4-mediated paravascular flow) uses CSF to clear amyloid-β, tau, and neuronal metabolic waste — disrupted in Alzheimer\'s disease and following TBI. CSF protein elevation (> 45 mg/dL) indicates blood–brain barrier breakdown, infection, or Guillain–Barré syndrome. xanthochromia (yellow CSF pigmentation from haemoglobin degradation products) is the gold-standard diagnosis for subarachnoid haemorrhage when CT is negative.',
      },
    ],
  },
];

// ── MEG Electrophysiology ──────────────────────────────────────────────────────

const megElectrophysiology: MegElectrophysiology = {
  bands: [
    {
      id: 'delta',
      label: 'Delta',
      freqRange: '< 4 Hz',
      freqMinHz: 0.5,
      freqMaxHz: 4,
      function: 'Deep NREM sleep, cortical down-states, focal pathology marker',
      detail:
        'Generated by large synchronised pyramidal-cell populations during NREM deep sleep (N3); thalamo-cortical hyperpolarisation drives cortical down-states (~0.75 Hz) characterised by complete cessation of membrane potential oscillations. Interspersed with brief up-states (depolarisation) creating slow oscillations (0.5–1 Hz) critical for hippocampus-to-cortex memory consolidation ("memory replay"). In the awake brain, focal delta activity (polymorphic delta activity, PDA) over a region indicates structural or vascular lesion (tumour, infarct, haematoma) — the underlying cortex is functionally disconnected from thalamic drive. Delta is also prominent in hepatic encephalopathy and diffuse TBI.',
    },
    {
      id: 'theta',
      label: 'Theta',
      freqRange: '4–8 Hz',
      freqMinHz: 4,
      freqMaxHz: 8,
      function: 'Hippocampal memory encoding, working memory carrier, rhythm entrainment',
      detail:
        'Hippocampal theta (generated by medial septal GABAergic pacemaker) coordinates memory encoding: CA1 pyramidal cells fire at specific theta phases encoding temporal order of events ("theta phase precession"). PFC–hippocampus theta coherence increases during working memory load and spatial navigation. Frontal midline theta power scales monotonically with cognitive difficulty; peak individual theta frequency (ITF) predicts working memory capacity. Theta-nested gamma coupling ("theta-gamma code") packages individual items in working memory: each ~25 ms gamma cycle encodes one item, fitting ~4–7 items within a single theta cycle — a neurophysiological account of Miller\'s 7 ± 2 limit.',
    },
    {
      id: 'alpha',
      label: 'Alpha',
      freqRange: '8–12 Hz',
      freqMinHz: 8,
      freqMaxHz: 12,
      function: 'Cortical inhibition gate, visual idle rhythm, thalamo-cortical pacemaker',
      detail:
        'Arises from intrinsic thalamic relay-cell burst oscillations resonating with cortico-thalamic feedback circuits. Occipital alpha (Berger\'s "Alphawellen", 8–10 Hz) is the "idle rhythm" of visual cortex: power inversely correlates with visual cortex excitability (alpha power = inhibition). Alpha suppression over a cortical region is the canonical indicator of task-related activation in MEG. Sensorimotor alpha (mu rhythm, 10–12 Hz) desynchronises bilaterally with movement preparation; asymmetric mu suppression is used to study lateralisation of motor control. Pre-stimulus alpha power over occipital cortex predicts perceptual detection of near-threshold visual stimuli.',
    },
    {
      id: 'beta',
      label: 'Beta',
      freqRange: '12–30 Hz',
      freqMinHz: 12,
      freqMaxHz: 30,
      function: 'Motor status-quo maintenance, top-down prediction, Parkinson\'s biomarker',
      detail:
        'Sensorimotor beta power maintains the current motor "status quo" — suppresses after movement onset and rebounds 500–1000 ms post-movement (post-movement beta rebound, PMBR). Long-range fronto-parietal beta synchrony (in the 18–25 Hz sub-band) transmits top-down predictions in predictive-coding frameworks; bottom-up gamma carries prediction errors. Excessive pathological beta oscillations (14–20 Hz) in cortico-basal-ganglia loops are the electrophysiological signature of Parkinson\'s disease — power in GPi LFPs correlates with bradykinesia severity. DBS at 130 Hz is thought to disrupt this pathological beta resonance by temporal decorrelation.',
    },
    {
      id: 'gamma',
      label: 'Gamma',
      freqRange: '≥ 30 Hz',
      freqMinHz: 30,
      freqMaxHz: null,
      function: 'Local E–I network computation, feature binding, attention-dependent',
      detail:
        'Driven by pyramidal-interneuron gamma (PING) mechanism: fast-spiking parvalbumin-positive (PV+) interneurons provide perisomatic inhibition to pyramidal cells, generating rhythmic population excitation at 30–80 Hz. Attention-dependent gamma power increases reflect "binding" of distributed feature representations — different visual features processed in different cortical areas are integrated by gamma-frequency synchrony. High-frequency broadband (HFB, 70–150 Hz), measured by intracranial EEG, is the closest proxy for local multiunit neural firing and is used in BCI spelling paradigms. Auditory steady-state response (ASSR) at 40 Hz is a classic MEG/EEG gamma entrainment paradigm; reduced 40 Hz ASSR is a robust biomarker in schizophrenia, indexing PV+ interneuron dysfunction.',
    },
  ],

  erp: {
    id: 'erp',
    label: 'Event-Related Potentials (ERPs)',
    description:
      'Time-locked averages of MEG/EEG signals that reveal the brain\'s sequential inference steps. Each ERP component indexes a specific computational stage in the brain\'s probabilistic model of the world.',
    waves: [
      {
        id: 'mmn',
        label: 'Mismatch Negativity (MMN)',
        latency: '100–250 ms post-deviant stimulus',
        distribution: '−1 to −5 µV; maximal frontocentral; bilateral temporal MEG sources',
        interpretation:
          'Automatic detection of a mismatch between the incoming stimulus and a learned auditory regularity (the brain\'s internal generative model of the acoustic environment). Occurs without attention: present during sleep, distraction, or in comatose patients. A clinical biomarker for recovery of consciousness after severe brain injury. Reduced MMN amplitude in schizophrenia indexes aberrant precision-weighting of auditory sensory predictions in the hierarchical predictive-coding model.',
        mathematicalBasis:
          'MMN amplitude ≈ |prediction − observation| — directly proportional to the magnitude of the prediction error signal generated in STG. Bayesian formulation: MMN indexes the precision-weighted prediction error ε = Σ⁻¹(observation − μ̂), where Σ is the uncertainty of the learned model and μ̂ is the model\'s expectation. In neural terms: superficial pyramidal cells in STG compute and broadcast ε upward to frontal cortex via fast feedforward (gamma) pathways.',
      },
      {
        id: 'p300',
        label: 'P300 (P3)',
        latency: '300–600 ms post-target; P3a ~250–280 ms (frontal), P3b ~300–500 ms (parietal)',
        distribution: 'P3a: frontocentral; P3b: centroparietal; amplitude 5–20 µV',
        interpretation:
          'P3a = involuntary attention capture / orienting response to task-irrelevant novel stimuli (frontal generators, active even without attention). P3b = voluntary context updating of an internal representation ("working model") following a task-relevant rare stimulus. P3b amplitude predicts subsequent memory — larger P300 to an event → better episodic recall. Prolonged P3b latency (> 380 ms) indexes slowed cognitive processing speed and is a sensitive biomarker for mild cognitive impairment, metabolic encephalopathies, and dementia.',
        mathematicalBasis:
          'P3b amplitude scales with Bayesian surprise: A(P3b) ∝ −log P(stimulus) — rare events (low prior probability) elicit larger P300s. Subjective probability (not physical rarity) is the governing variable: a "rare" deviant that the participant has adapted to elicits smaller P300. In predictive-coding terms, the P300 reflects the deep-layer (prediction) update step rather than the superficial-layer (error) step that drives the MMN — the brain revises its priors after receiving the MMN-signalled error.',
      },
    ],
  },
};

// ── fMRI Haemodynamics ─────────────────────────────────────────────────────────

const fmriHemodynamics: FmriHemodynamics = {
  id: 'hrf',
  label: 'Hemodynamic Response Function (HRF)',
  function: 'Neurovascular coupling — temporal smearing of neural events into BOLD signal',
  detail:
    'Neural firing → glutamate release → astrocyte uptake → vasoactive signals (prostaglandins, nitric oxide, K⁺) via astrocyte end-feet → pial arteriole dilation → capillary bed O₂ delivery increase → reduced deoxyhaemoglobin → increased T2* BOLD MR signal. The HRF is the vascular convolution kernel that transforms neural activity into the measurable BOLD response: BOLD(t) = neural(t) * HRF(t). Canonical HRF parameters: response onset 1–2 s post-stimulus; peak 4–6 s; undershoot 12–20 s (sustained elevated cerebral blood volume after flow normalises). Design implication: any MEG transient — such as a 150 ms MMN or a 50 ms auditory onset response — appears as a ~10–15 s BOLD blob in fMRI; rapid-presentation event-related designs require HRF deconvolution (e.g. finite impulse response, FIR) to recover neural onset timing from haemodynamic responses. The 4–6 s delay means that pairing MEG (ms resolution) with fMRI (mm spatial precision) in the multimodal workspace recovers both when and where neural computations occur.',
};

// ── Root export ────────────────────────────────────────────────────────────────

/**
 * The complete neurological reference dictionary.
 * Import this in ReferencePanelContext and ReferenceDrawer.
 */
export const neuroDictionary: NeuroDictionary = {
  anatomy:               anatomyGroups,
  megElectrophysiology:  megElectrophysiology,
  fmriHemodynamics:      fmriHemodynamics,
};

// ── Label → region lookup builder ─────────────────────────────────────────────

/** Maps every FreeSurfer label ID to the group and region that owns it. */
export interface LabelNavTarget {
  groupId:  string;
  regionId: string;
}

/**
 * Build a lookup map from FreeSurfer label ID → {groupId, regionId}.
 * Calling navigateToRegion(label) in ReferencePanelContext uses this map to
 * jump the drawer directly to the correct accordion entry.
 */
export function buildLabelLookup(): Map<number, LabelNavTarget> {
  const map = new Map<number, LabelNavTarget>();
  for (const group of neuroDictionary.anatomy) {
    for (const region of group.regions) {
      for (const label of region.fsLabels) {
        map.set(label, { groupId: group.id, regionId: region.id });
      }
    }
  }
  return map;
}
