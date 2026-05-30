//! Dojo Musician Rooms — musicians as Plato rooms
//!
//! Each instrument is a room with vibe embeddings that generate musical output.
//! Rooms murmur to stay in sync, forming a cellular graph band.

use std::collections::HashMap;

// ── Core Types ──────────────────────────────────────────────────────────────

/// Room identifier in the cellular graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoomId(pub u64);

/// A musician's style encoded as an embedding.
/// Dimensions: [dark, bright, warm, harsh, dense, sparse, fast, slow, dry, wet, tight, loose, forward, distant, smooth, rough]
#[derive(Debug, Clone)]
pub struct MusicianVibe {
    pub embedding: [f64; 16],
    pub instrument: Instrument,
    pub name: String,
}

/// Instrument types in the band
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instrument {
    Bass,
    Drums,
    Melody,
    Pads,
    FX,
    Keys,
    Guitar,
    Strings,
}

/// A single note in a pattern
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub pitch: u8,
    pub start_tick: u32,
    pub duration_ticks: u32,
    pub velocity: u8,
}

/// A musical pattern generated from a vibe
#[derive(Debug, Clone)]
pub struct Pattern {
    pub notes: Vec<Note>,
    pub velocity_curve: Vec<u8>,
    pub timing_offsets: Vec<f64>,
    pub duration_ticks: u32,
}

/// Musical scale
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    Major,
    Minor,
    Dorian,
    Mixolydian,
    PentatonicMinor,
    PentatonicMajor,
    Blues,
    Chromatic,
}

impl Scale {
    /// Return semitone intervals from root for this scale
    pub fn intervals(&self) -> &'static [u8] {
        match self {
            Scale::Major => &[0, 2, 4, 5, 7, 9, 11],
            Scale::Minor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            Scale::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            Scale::PentatonicMinor => &[0, 3, 5, 7, 10],
            Scale::PentatonicMajor => &[0, 2, 4, 7, 9],
            Scale::Blues => &[0, 3, 5, 6, 7, 10],
            Scale::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    }

    /// Check if a pitch is in the given key + scale
    pub fn contains_pitch(&self, key: u8, pitch: u8) -> bool {
        if *self == Scale::Chromatic {
            return true;
        }
        let interval = (pitch as i32 - key as i32).rem_euclid(12) as u8;
        self.intervals().contains(&interval)
    }
}

/// A tick of information from another room
#[derive(Debug, Clone, PartialEq)]
pub struct Tick {
    pub source: RoomId,
    pub pattern_summary: PatternSummary,
    pub timestamp: u64,
}

/// Summary of a pattern sent via murmur
#[derive(Debug, Clone, PartialEq)]
pub struct PatternSummary {
    pub note_count: usize,
    pub density: f64,
    pub avg_velocity: f64,
    pub instrument: Instrument,
}

/// A murmur message between rooms
#[derive(Debug, Clone)]
pub struct Murmur {
    pub from: RoomId,
    pub ticks: Vec<Tick>,
}

/// Jam session context
#[derive(Debug, Clone)]
pub struct JamSession {
    pub bpm: f64,
    pub key: u8,
    pub scale: Scale,
    pub bars: u32,
    pub current_bar: u32,
    pub energy: f64,
}

/// Context passed to pattern generation
#[derive(Debug, Clone)]
pub struct JamContext {
    pub bpm: f64,
    pub key: u8,
    pub scale: Scale,
    pub bars: u32,
    pub energy: f64,
}

impl From<&JamSession> for JamContext {
    fn from(s: &JamSession) -> Self {
        JamContext {
            bpm: s.bpm,
            key: s.key,
            scale: s.scale,
            bars: s.bars,
            energy: s.energy,
        }
    }
}

/// A musician room in the cellular graph
pub struct MusicianRoom {
    pub musician: MusicianVibe,
    pub perception_db: Vec<Tick>,
    pub prediction_db: Vec<Tick>,
    pub current_pattern: Pattern,
    pub groove_lock: Option<RoomId>,
    pub key_center: u8,
    pub scale: Scale,
    pub bpm: f64,
    pub id: RoomId,
}

/// The dojo — a collection of musician rooms playing together
pub struct Dojo {
    pub rooms: HashMap<RoomId, MusicianRoom>,
    pub graph: CellularGraph,
    pub jam_session: Option<JamSession>,
    pub next_id: u64,
}

/// Simple cellular graph tracking connections between rooms
#[derive(Debug, Clone)]
pub struct CellularGraph {
    pub edges: Vec<(RoomId, RoomId)>,
}

impl CellularGraph {
    pub fn new() -> Self {
        CellularGraph { edges: Vec::new() }
    }

    pub fn connect(&mut self, a: RoomId, b: RoomId) {
        if !self.edges.contains(&(a, b)) && !self.edges.contains(&(b, a)) {
            self.edges.push((a, b));
        }
    }

    pub fn neighbors(&self, id: RoomId) -> Vec<RoomId> {
        let mut neighbors = Vec::new();
        for (a, b) in &self.edges {
            if *a == id {
                neighbors.push(*b);
            } else if *b == id {
                neighbors.push(*a);
            }
        }
        neighbors
    }
}

// ── MusicianVibe helpers ────────────────────────────────────────────────────

impl MusicianVibe {
    pub fn new(instrument: Instrument, name: &str) -> Self {
        let mut embedding = [0.5; 16];
        // Default vibe per instrument
        match instrument {
            Instrument::Bass => {
                embedding[0] = 0.6; // darker
                embedding[2] = 0.5; // warm
                embedding[5] = 0.7; // sparse
                embedding[10] = 0.8; // tight
            }
            Instrument::Drums => {
                embedding[14] = 0.6; // smooth
                embedding[10] = 0.7; // tight
                embedding[4] = 0.6; // dense
            }
            Instrument::Melody => {
                embedding[1] = 0.6; // bright
                embedding[6] = 0.6; // fast
                embedding[12] = 0.7; // forward
            }
            Instrument::Pads => {
                embedding[2] = 0.7; // warm
                embedding[7] = 0.6; // slow
                embedding[9] = 0.7; // wet
                embedding[5] = 0.6; // sparse
            }
            Instrument::FX => {
                embedding[3] = 0.5; // harsh
                embedding[9] = 0.8; // wet
                embedding[11] = 0.6; // loose
            }
            Instrument::Keys => {
                embedding[1] = 0.6; // bright
                embedding[14] = 0.7; // smooth
                embedding[10] = 0.7; // tight
            }
            Instrument::Guitar => {
                embedding[2] = 0.6; // warm
                embedding[15] = 0.5; // rough
                embedding[12] = 0.6; // forward
            }
            Instrument::Strings => {
                embedding[2] = 0.7; // warm
                embedding[14] = 0.7; // smooth
                embedding[9] = 0.5; // wet
            }
        }
        MusicianVibe {
            embedding,
            instrument,
            name: name.to_string(),
        }
    }

    /// Density derived from dense (dim4) vs sparse (dim5)
    pub fn density(&self) -> f64 {
        self.embedding[4] - self.embedding[5] + 0.5
    }

    /// Looseness from loose dimension (dim11)
    pub fn looseness(&self) -> f64 {
        self.embedding[11]
    }

    /// Roughness from rough dimension (dim15)
    pub fn roughness(&self) -> f64 {
        self.embedding[15]
    }

    /// Brightness from dark (dim0) and bright (dim1)
    pub fn brightness(&self) -> f64 {
        self.embedding[1] - self.embedding[0] + 0.5
    }

    /// Determine pitch range from brightness and context
    pub fn brightness_range(&self, _ctx: &JamContext) -> (u8, u8) {
        let b = self.brightness();
        let base = match self.instrument {
            Instrument::Bass => 28u8,
            Instrument::Drums => 36,
            _ => 48,
        };
        let range_spread = 24u8;
        let offset = (b * 12.0) as u8;
        (base + offset, base + offset + range_spread)
    }

    /// Normalize embedding so no dimension exceeds [0, 1]
    pub fn normalize(&mut self) {
        for d in &mut self.embedding {
            *d = d.clamp(0.0, 1.0);
        }
    }
}

// ── MusicianRoom impl ───────────────────────────────────────────────────────

impl MusicianRoom {
    /// Create a new musician room
    pub fn new(id: RoomId, instrument: Instrument, name: &str) -> Self {
        let vibe = MusicianVibe::new(instrument, name);
        MusicianRoom {
            musician: vibe,
            perception_db: Vec::new(),
            prediction_db: Vec::new(),
            current_pattern: Pattern::empty(96),
            groove_lock: None,
            key_center: 60,
            scale: Scale::Minor,
            bpm: 120.0,
            id,
        }
    }

    /// Generate a pattern from the current vibe embedding
    pub fn generate(&self, bars: u32, context: &JamContext) -> Pattern {
        generate_from_vibe(&self.musician, bars, context)
    }

    /// Adjust vibe based on qualitative description
    pub fn adjust_vibe(&mut self, description: &str, delta: f64) {
        let (embed_delta, _strength) = description_to_delta(description);
        for i in 0..16 {
            self.musician.embedding[i] += embed_delta[i] * delta;
        }
        self.musician.normalize();
    }

    /// Lock groove to another room
    pub fn lock_groove(&mut self, other: RoomId) {
        self.groove_lock = Some(other);
    }

    /// Listen to what other rooms are playing (murmur)
    pub fn listen(&mut self, murmur: Murmur) {
        for tick in murmur.ticks {
            self.perception_db.push(tick.clone());
            // Predict that similar ticks will continue
            let prediction = Tick {
                source: murmur.from,
                timestamp: tick.timestamp + 96,
                pattern_summary: tick.pattern_summary.clone(),
            };
            self.prediction_db.push(prediction);
        }
    }

    /// React to what was heard — adjust pattern
    pub fn react(&mut self) -> Pattern {
        if self.perception_db.is_empty() {
            return self.current_pattern.clone();
        }

        // Adjust density based on what others are doing
        let avg_density: f64 = self
            .perception_db
            .iter()
            .map(|t| t.pattern_summary.density)
            .sum::<f64>()
            / self.perception_db.len() as f64;

        // If others are dense, go sparser; if sparse, fill in
        let density_shift = 0.5 - avg_density;
        self.musician.embedding[4] += density_shift * 0.1;
        self.musician.embedding[5] -= density_shift * 0.1;
        self.musician.normalize();

        let ctx = JamContext {
            bpm: self.bpm,
            key: self.key_center,
            scale: self.scale,
            bars: 4,
            energy: 0.5,
        };
        let pattern = self.generate(4, &ctx);
        self.current_pattern = pattern.clone();
        pattern
    }

    /// Double-entry check: perception events = prediction events
    pub fn balance_check(&self) -> bool {
        self.perception_db.len() == self.prediction_db.len()
    }
}

// ── Pattern helpers ─────────────────────────────────────────────────────────

impl Pattern {
    pub fn empty(ticks: u32) -> Self {
        Pattern {
            notes: Vec::new(),
            velocity_curve: Vec::new(),
            timing_offsets: Vec::new(),
            duration_ticks: ticks,
        }
    }
}

// ── Pattern Generation Algorithm ────────────────────────────────────────────

/// Generate musical patterns FROM vibe embeddings
pub fn generate_from_vibe(vibe: &MusicianVibe, bars: u32, ctx: &JamContext) -> Pattern {
    let ticks_per_bar = 96u32;
    let total_ticks = bars * ticks_per_bar;

    // 1. Determine note density from vibe
    let density = vibe.density().clamp(0.1, 1.0);
    let energy = ctx.energy.clamp(0.0, 1.0);
    let base_notes = bars * 4;
    let note_count = ((base_notes as f64) * density * (0.5 + energy * 0.5)).max(bars as f64).ceil() as usize;

    // 2. Determine rhythm from vibe
    let looseness = vibe.looseness();
    let roughness = vibe.roughness();
    let swing = looseness * 0.3;
    let humanize = roughness * 0.05; // max 50ms

    // 3. Determine pitch selection
    let (low, high) = vibe.brightness_range(ctx);

    // Build scale notes in range
    let scale_notes: Vec<u8> = (low..=high)
        .filter(|&p| ctx.scale.contains_pitch(ctx.key, p))
        .collect();

    if scale_notes.is_empty() {
        return Pattern::empty(total_ticks);
    }

    // 4. Generate notes
    let mut notes = Vec::with_capacity(note_count);
    let tick_spacing = total_ticks / note_count.max(1) as u32;

    // Velocity shaping based on roughness
    let base_vel = ((energy * 80.0 + 40.0) as u8).min(127);
    let vel_variance = (roughness * 30.0) as u8;

    let mut velocity_curve = Vec::with_capacity(note_count);
    let mut timing_offsets = Vec::with_capacity(note_count);

    // Simple pseudo-random using position (deterministic from vibe)
    for i in 0..note_count {
        let tick_pos = (i as u32 * tick_spacing) % total_ticks;

        // Apply swing to off-beats
        let is_offbeat = (tick_pos / (ticks_per_bar / 8)) % 2 == 1;
        let swing_offset = if is_offbeat { swing * (ticks_per_bar as f64 / 8.0) } else { 0.0 };

        // Humanize offset in seconds (converted to approximate ticks later)
        let human_offset = humanize * (((i * 7 + 3) % 11) as f64 / 10.0 - 0.5);
        timing_offsets.push(human_offset);

        // Pick a note from scale
        let note_idx = ((i * 5 + (vibe.embedding[0] * 7.0) as usize) % scale_notes.len().max(1)).min(scale_notes.len() - 1);
        let pitch = scale_notes[note_idx];

        // Velocity with variation
        let vel_mod = ((i as i32 * 3 + 1) % (vel_variance as i32 + 1)).abs() as u8;
        let velocity = (base_vel as u16 + vel_mod as u16).min(127) as u8;

        let start_tick = (tick_pos as f64 + swing_offset) as u32;

        // Duration: instruments have different typical durations
        let dur = match vibe.instrument {
            Instrument::Drums => ticks_per_bar / 8,
            Instrument::Bass => ticks_per_bar / 2,
            Instrument::Pads => ticks_per_bar * 2,
            _ => ticks_per_bar / 4,
        };

        notes.push(Note {
            pitch,
            start_tick: start_tick.min(total_ticks - 1),
            duration_ticks: dur.min(total_ticks - start_tick),
            velocity,
        });
        velocity_curve.push(velocity);
    }

    Pattern {
        notes,
        velocity_curve,
        timing_offsets,
        duration_ticks: total_ticks,
    }
}

// ── Vibe-to-Description Mapping ─────────────────────────────────────────────

/// Maps qualitative descriptions to embedding deltas.
/// Returns (embedding delta, strength).
pub fn description_to_delta(desc: &str) -> ([f64; 16], f64) {
    let desc_lower = desc.to_lowercase();
    let mut delta = [0.0f64; 16];

    if desc_lower.contains("dark") {
        delta[0] += 0.2;  // more dark
        delta[1] -= 0.2;  // less bright
    }
    if desc_lower.contains("bright") {
        delta[1] += 0.2;
        delta[0] -= 0.2;
    }
    if desc_lower.contains("warm") {
        delta[2] += 0.3;
    }
    if desc_lower.contains("harsh") {
        delta[3] += 0.2;
    }
    if desc_lower.contains("dense") {
        delta[4] += 0.2;
        delta[5] -= 0.1;
    }
    if desc_lower.contains("sparse") {
        delta[5] += 0.2;
        delta[4] -= 0.1;
    }
    if desc_lower.contains("fast") {
        delta[6] += 0.2;
        delta[7] -= 0.1;
    }
    if desc_lower.contains("slow") {
        delta[7] += 0.2;
        delta[6] -= 0.1;
    }
    if desc_lower.contains("dry") {
        delta[8] += 0.2;
        delta[9] -= 0.1;
    }
    if desc_lower.contains("wet") {
        delta[9] += 0.2;
        delta[8] -= 0.1;
    }
    if desc_lower.contains("tight") {
        delta[10] += 0.2;
        delta[11] -= 0.1;
    }
    if desc_lower.contains("loose") || desc_lower.contains("looser") {
        delta[11] += 0.2;
        delta[10] -= 0.1;
    }
    if desc_lower.contains("forward") {
        delta[12] += 0.2;
        delta[13] -= 0.1;
    }
    if desc_lower.contains("distant") {
        delta[13] += 0.2;
        delta[12] -= 0.1;
    }
    if desc_lower.contains("smooth") {
        delta[14] += 0.2;
        delta[15] -= 0.1;
    }
    if desc_lower.contains("rough") {
        delta[15] += 0.2;
        delta[14] -= 0.1;
    }
    if desc_lower.contains("underwater") {
        delta[0] += 0.3;
        delta[1] -= 0.2;
        delta[2] += 0.1;
        delta[3] -= 0.1;
        delta[4] -= 0.2;
        delta[5] += 0.1;
        delta[8] -= 0.1;
        delta[9] += 0.3;
        delta[12] -= 0.1;
        delta[13] += 0.2;
        delta[14] -= 0.1;
    }
    if desc_lower.contains("wall") || desc_lower.contains("through a wall") {
        delta[12] -= 0.2;
        delta[5] += 0.2;
        delta[13] += 0.2;
        delta[9] += 0.1;
    }

    let has_changes = delta.iter().any(|&d| d != 0.0);
    (delta, if has_changes { 0.5 } else { 0.0 })
}

// ── Dojo impl ───────────────────────────────────────────────────────────────

impl Dojo {
    /// Create a new empty dojo
    pub fn new() -> Self {
        Dojo {
            rooms: HashMap::new(),
            graph: CellularGraph::new(),
            jam_session: None,
            next_id: 0,
        }
    }

    /// Create a band with specified instruments
    pub fn new_band(instruments: &[Instrument]) -> Self {
        let mut dojo = Dojo::new();
        let ids: Vec<RoomId> = instruments
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let id = RoomId(i as u64);
                let name = format!("{:?}", inst);
                let room = MusicianRoom::new(id, *inst, &name);
                dojo.rooms.insert(id, room);
                id
            })
            .collect();

        // Connect all rooms in the graph
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                dojo.graph.connect(ids[i], ids[j]);
            }
        }

        dojo.jam_session = Some(JamSession {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            current_bar: 0,
            energy: 0.5,
        });

        dojo
    }

    fn alloc_id(&mut self) -> RoomId {
        let id = RoomId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a room to the dojo
    pub fn add_room(&mut self, instrument: Instrument, name: &str) -> RoomId {
        let id = self.alloc_id();
        let room = MusicianRoom::new(id, instrument, name);
        self.rooms.insert(id, room);
        id
    }

    /// Run one iteration: generate, murmur, react, mix
    pub fn iterate(&mut self) -> Vec<Pattern> {
        let ctx = self.jam_session.as_ref().map(|s| JamContext::from(s)).unwrap_or(JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        });

        // Generate patterns for all rooms
        let patterns: HashMap<RoomId, Pattern> = self
            .rooms
            .iter()
            .map(|(id, room)| (*id, room.generate(ctx.bars, &ctx)))
            .collect();

        // Create murmurs
        let murmurs: HashMap<RoomId, Murmur> = self
            .rooms
            .keys()
            .map(|&id| {
                let neighbors = self.graph.neighbors(id);
                let ticks: Vec<Tick> = neighbors
                    .iter()
                    .filter_map(|&n_id| {
                        patterns.get(&n_id).map(|p| Tick {
                            source: n_id,
                            pattern_summary: PatternSummary {
                                note_count: p.notes.len(),
                                density: if p.notes.is_empty() {
                                    0.0
                                } else {
                                    p.notes.len() as f64 / (p.duration_ticks as f64 / 96.0)
                                },
                                avg_velocity: if p.notes.is_empty() {
                                    0.0
                                } else {
                                    p.notes.iter().map(|n| n.velocity as f64).sum::<f64>()
                                        / p.notes.len() as f64
                                },
                                instrument: self.rooms.get(&n_id).map(|r| r.musician.instrument).unwrap_or(Instrument::Melody),
                            },
                            timestamp: 0,
                        })
                    })
                    .collect();
                (id, Murmur { from: id, ticks })
            })
            .collect();

        // Deliver murmurs and let rooms react
        let mut result = Vec::new();
        let room_ids: Vec<RoomId> = self.rooms.keys().copied().collect();
        for id in room_ids {
            if let Some(murmur) = murmurs.get(&id).cloned() {
                if let Some(room) = self.rooms.get_mut(&id) {
                    room.listen(murmur);
                    let pattern = room.react();
                    result.push(pattern);
                }
            }
        }

        result
    }

    /// Apply user feedback as vibe adjustments
    pub fn shape(&mut self, room_id: RoomId, description: &str, strength: f64) {
        if let Some(room) = self.rooms.get_mut(&room_id) {
            room.adjust_vibe(description, strength);
        }
    }

    /// Export all patterns as MIDI (basic MIDI file format)
    pub fn to_midi(&self) -> Vec<u8> {
        let mut midi = Vec::new();

        // MIDI header: MThd
        midi.extend_from_slice(b"MThd");
        // Header length (6 bytes)
        midi.extend_from_slice(&6u32.to_be_bytes());
        // Format 0 (single track)
        midi.extend_from_slice(&0u16.to_be_bytes());
        // Number of tracks
        midi.extend_from_slice(&1u16.to_be_bytes());
        // Ticks per quarter note (96 = 96 PPQ)
        midi.extend_from_slice(&96u16.to_be_bytes());

        // Track header
        midi.extend_from_slice(b"MTrk");
        let track_len_pos = midi.len();
        midi.extend_from_slice(&0u32.to_be_bytes()); // placeholder

        // Collect all notes from all rooms, sorted by start_tick
        let mut all_notes: Vec<&Note> = Vec::new();
        for room in self.rooms.values() {
            for note in &room.current_pattern.notes {
                all_notes.push(note);
            }
        }
        all_notes.sort_by_key(|n| n.start_tick);

        // Write events as delta-time + note on/off
        let mut last_tick: u32 = 0;
        for note in &all_notes {
            // Delta time (variable length)
            let delta = note.start_tick.saturating_sub(last_tick);
            write_var_len(&mut midi, delta);
            // Note on: status byte (0x90 = channel 1), note, velocity
            midi.push(0x90);
            midi.push(note.pitch.min(127));
            midi.push(note.velocity.min(127));

            // Note off at end of note
            write_var_len(&mut midi, note.duration_ticks);
            // Note off: 0x80, note, velocity 0
            midi.push(0x80);
            midi.push(note.pitch.min(127));
            midi.push(0);

            last_tick = note.start_tick + note.duration_ticks;
        }

        // End of track
        write_var_len(&mut midi, 0);
        midi.push(0xFF);
        midi.push(0x2F);
        midi.push(0x00);

        // Fill in track length
        let track_len = (midi.len() - track_len_pos - 4) as u32;
        midi[track_len_pos..track_len_pos + 4].copy_from_slice(&track_len.to_be_bytes());

        midi
    }

    /// Get the vibe of the whole band (average embedding)
    pub fn band_vibe(&self) -> [f64; 16] {
        if self.rooms.is_empty() {
            return [0.5; 16];
        }
        let mut avg = [0.0; 16];
        for room in self.rooms.values() {
            for (i, &v) in room.musician.embedding.iter().enumerate() {
                avg[i] += v;
            }
        }
        let count = self.rooms.len() as f64;
        for d in &mut avg {
            *d /= count;
        }
        avg
    }
}

/// Write a variable-length quantity (MIDI standard)
fn write_var_len(buf: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        buf.push(0);
        return;
    }
    let mut bytes = Vec::new();
    while value > 0 {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if !bytes.is_empty() {
            byte |= 0x80;
        }
        bytes.push(byte);
    }
    bytes.reverse();
    buf.extend_from_slice(&bytes);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1_create_musician_room_with_vibe() {
        let room = MusicianRoom::new(RoomId(0), Instrument::Bass, "Funky Bass");
        assert_eq!(room.musician.instrument, Instrument::Bass);
        assert_eq!(room.musician.name, "Funky Bass");
        // Embedding should be in [0, 1]
        for &d in &room.musician.embedding {
            assert!(d >= 0.0 && d <= 1.0, "embedding dimension out of range: {}", d);
        }
    }

    #[test]
    fn test_2_generate_pattern_returns_valid_notes() {
        let room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        };
        let pattern = room.generate(4, &ctx);
        assert!(!pattern.notes.is_empty(), "should generate some notes");
        assert_eq!(pattern.duration_ticks, 4 * 96);
        for note in &pattern.notes {
            assert!(note.velocity <= 127);
            assert!(note.start_tick < pattern.duration_ticks);
        }
    }

    #[test]
    fn test_3_notes_are_in_key() {
        let room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        };
        let pattern = room.generate(4, &ctx);
        for note in &pattern.notes {
            assert!(
                ctx.scale.contains_pitch(ctx.key, note.pitch),
                "note {} not in key {} scale {:?}",
                note.pitch,
                ctx.key,
                ctx.scale
            );
        }
    }

    #[test]
    fn test_4_note_density_matches_vibe() {
        let mut dense_vibe = MusicianVibe::new(Instrument::Drums, "Dense");
        dense_vibe.embedding[4] = 0.9; // dense
        dense_vibe.embedding[5] = 0.1; // not sparse

        let mut sparse_vibe = MusicianVibe::new(Instrument::Drums, "Sparse");
        sparse_vibe.embedding[4] = 0.1;
        sparse_vibe.embedding[5] = 0.9;

        let ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        };
        let dense_pattern = generate_from_vibe(&dense_vibe, 4, &ctx);
        let sparse_pattern = generate_from_vibe(&sparse_vibe, 4, &ctx);
        assert!(
            dense_pattern.notes.len() > sparse_pattern.notes.len(),
            "dense vibe ({}) should have more notes than sparse ({})",
            dense_pattern.notes.len(),
            sparse_pattern.notes.len()
        );
    }

    #[test]
    fn test_5_swing_matches_looseness() {
        let mut tight_vibe = MusicianVibe::new(Instrument::Bass, "Tight");
        tight_vibe.embedding[11] = 0.0; // not loose

        let mut loose_vibe = MusicianVibe::new(Instrument::Bass, "Loose");
        loose_vibe.embedding[11] = 1.0;

        assert!(tight_vibe.looseness() < loose_vibe.looseness());
        // Swing is derived from looseness
        let tight_swing = tight_vibe.looseness() * 0.3;
        let loose_swing = loose_vibe.looseness() * 0.3;
        assert!(tight_swing < loose_swing);
    }

    #[test]
    fn test_6_adjust_vibe_darker() {
        let mut room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let dark_before = room.musician.embedding[0];
        let bright_before = room.musician.embedding[1];
        room.adjust_vibe("darker", 1.0);
        assert!(
            room.musician.embedding[0] > dark_before,
            "dark dimension should increase"
        );
        assert!(
            room.musician.embedding[1] < bright_before,
            "bright dimension should decrease"
        );
    }

    #[test]
    fn test_7_adjust_vibe_underwater() {
        let mut room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let dark_before = room.musician.embedding[0];
        let wet_before = room.musician.embedding[9];
        room.adjust_vibe("underwater", 1.0);
        assert!(
            room.musician.embedding[0] > dark_before,
            "underwater should increase dark"
        );
        assert!(
            room.musician.embedding[9] > wet_before,
            "underwater should increase wet"
        );
    }

    #[test]
    fn test_8_lock_groove() {
        let mut room1 = MusicianRoom::new(RoomId(0), Instrument::Drums, "Drums");
        let mut room2 = MusicianRoom::new(RoomId(1), Instrument::Bass, "Bass");
        assert!(room1.groove_lock.is_none());
        room1.lock_groove(RoomId(1));
        assert_eq!(room1.groove_lock, Some(RoomId(1)));
        room2.lock_groove(RoomId(0));
        assert_eq!(room2.groove_lock, Some(RoomId(0)));
    }

    #[test]
    fn test_9_listen_and_react() {
        let mut room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let murmur = Murmur {
            from: RoomId(1),
            ticks: vec![Tick {
                source: RoomId(1),
                pattern_summary: PatternSummary {
                    note_count: 16,
                    density: 0.5,
                    avg_velocity: 100.0,
                    instrument: Instrument::Drums,
                },
                timestamp: 0,
            }],
        };
        room.listen(murmur);
        assert_eq!(room.perception_db.len(), 1);
        assert_eq!(room.prediction_db.len(), 1);

        let pattern = room.react();
        assert!(!pattern.notes.is_empty());
    }

    #[test]
    fn test_10_balance_check() {
        let mut room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        assert!(room.balance_check()); // both empty

        let murmur = Murmur {
            from: RoomId(1),
            ticks: vec![Tick {
                source: RoomId(1),
                pattern_summary: PatternSummary {
                    note_count: 8,
                    density: 0.4,
                    avg_velocity: 80.0,
                    instrument: Instrument::Bass,
                },
                timestamp: 0,
            }],
        };
        room.listen(murmur);
        assert!(room.balance_check(), "perception count should equal prediction count");
    }

    #[test]
    fn test_11_dojo_new_band() {
        let dojo = Dojo::new_band(&[Instrument::Bass, Instrument::Drums, Instrument::Melody]);
        assert_eq!(dojo.rooms.len(), 3);
        assert!(dojo.jam_session.is_some());
        assert_eq!(dojo.graph.edges.len(), 3); // fully connected 3 nodes
    }

    #[test]
    fn test_12_dojo_iterate() {
        let mut dojo = Dojo::new_band(&[Instrument::Bass, Instrument::Drums, Instrument::Melody]);
        let patterns = dojo.iterate();
        assert_eq!(patterns.len(), 3, "should produce one pattern per room");
    }

    #[test]
    fn test_13_dojo_shape() {
        let mut dojo = Dojo::new_band(&[Instrument::Bass, Instrument::Drums]);
        let room_id = RoomId(0);
        let dark_before = dojo.rooms[&room_id].musician.embedding[0];
        dojo.shape(room_id, "darker", 1.0);
        let dark_after = dojo.rooms[&room_id].musician.embedding[0];
        assert!(dark_after > dark_before, "shape should adjust vibe");
    }

    #[test]
    fn test_14_band_vibe() {
        let dojo = Dojo::new_band(&[Instrument::Bass, Instrument::Drums, Instrument::Melody]);
        let vibe = dojo.band_vibe();
        // Average of 3 rooms, each dimension should be between 0 and 1
        for &d in &vibe {
            assert!(d >= 0.0 && d <= 1.0);
        }
    }

    #[test]
    fn test_15_different_vibes_different_patterns() {
        let ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        };
        let mut vibe1 = MusicianVibe::new(Instrument::Bass, "Bass 1");
        vibe1.embedding[0] = 0.9; // very dark
        let mut vibe2 = MusicianVibe::new(Instrument::Bass, "Bass 2");
        vibe2.embedding[1] = 0.9; // very bright

        let p1 = generate_from_vibe(&vibe1, 4, &ctx);
        let p2 = generate_from_vibe(&vibe2, 4, &ctx);

        // Different vibes should produce different note sets
        let pitches1: Vec<u8> = p1.notes.iter().map(|n| n.pitch).collect();
        let pitches2: Vec<u8> = p2.notes.iter().map(|n| n.pitch).collect();
        assert_ne!(pitches1, pitches2, "different vibes should produce different patterns");
    }

    #[test]
    fn test_16_groove_locked_timing() {
        let mut room1 = MusicianRoom::new(RoomId(0), Instrument::Drums, "Drums");
        let mut room2 = MusicianRoom::new(RoomId(1), Instrument::Bass, "Bass");
        room1.lock_groove(RoomId(1));
        room2.lock_groove(RoomId(0));

        // Both groove-locked rooms should have the lock set
        assert_eq!(room1.groove_lock, Some(RoomId(1)));
        assert_eq!(room2.groove_lock, Some(RoomId(0)));
    }

    #[test]
    fn test_17_midi_export() {
        let mut dojo = Dojo::new_band(&[Instrument::Bass, Instrument::Drums]);
        // Generate patterns first
        let ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        };
        for room in dojo.rooms.values_mut() {
            room.current_pattern = room.generate(4, &ctx);
        }
        let midi = dojo.to_midi();
        // Should start with MThd header
        assert_eq!(&midi[0..4], b"MThd");
        // Should contain MTrk
        assert!(midi.windows(4).any(|w| w == b"MTrk"));
        // Should be non-trivial length
        assert!(midi.len() > 20);
    }

    #[test]
    fn test_18_embedding_stays_normalized() {
        let mut room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        // Apply extreme adjustments
        for _ in 0..10 {
            room.adjust_vibe("darker brighter warmer harsher", 2.0);
        }
        for &d in &room.musician.embedding {
            assert!(d >= 0.0 && d <= 1.0, "embedding should stay normalized: {}", d);
        }
    }

    #[test]
    fn test_19_murmurs_carry_context() {
        let mut room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let murmur = Murmur {
            from: RoomId(1),
            ticks: vec![
                Tick {
                    source: RoomId(1),
                    pattern_summary: PatternSummary {
                        note_count: 12,
                        density: 0.6,
                        avg_velocity: 90.0,
                        instrument: Instrument::Drums,
                    },
                    timestamp: 100,
                },
                Tick {
                    source: RoomId(2),
                    pattern_summary: PatternSummary {
                        note_count: 4,
                        density: 0.2,
                        avg_velocity: 70.0,
                        instrument: Instrument::Bass,
                    },
                    timestamp: 100,
                },
            ],
        };
        room.listen(murmur);
        assert_eq!(room.perception_db.len(), 2);
        assert_eq!(room.prediction_db.len(), 2);
        assert_eq!(room.perception_db[0].source, RoomId(1));
        assert_eq!(room.perception_db[1].pattern_summary.instrument, Instrument::Bass);
    }

    #[test]
    fn test_20_energy_scales_intensity() {
        let vibe = MusicianVibe::new(Instrument::Drums, "Drums");

        let low_energy_ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.1,
        };
        let high_energy_ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.9,
        };

        let low = generate_from_vibe(&vibe, 4, &low_energy_ctx);
        let high = generate_from_vibe(&vibe, 4, &high_energy_ctx);

        let avg_vel_low: f64 = if low.notes.is_empty() { 0.0 } else { low.notes.iter().map(|n| n.velocity as f64).sum::<f64>() / low.notes.len() as f64 };
        let avg_vel_high: f64 = if high.notes.is_empty() { 0.0 } else { high.notes.iter().map(|n| n.velocity as f64).sum::<f64>() / high.notes.len() as f64 };

        assert!(
            avg_vel_high > avg_vel_low,
            "high energy ({}) should have higher velocity than low ({})",
            avg_vel_high,
            avg_vel_low
        );
    }

    #[test]
    fn test_21_scale_constraint() {
        let scales = [Scale::Major, Scale::Minor, Scale::Dorian, Scale::PentatonicMinor, Scale::Blues];
        for scale in scales {
            for key in [0u8, 3, 7] {
                let room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Test");
                let ctx = JamContext {
                    bpm: 120.0,
                    key,
                    scale,
                    bars: 4,
                    energy: 0.5,
                };
                let pattern = room.generate(4, &ctx);
                for note in &pattern.notes {
                    assert!(
                        scale.contains_pitch(key, note.pitch),
                        "pitch {} not in scale {:?} key {}",
                        note.pitch,
                        scale,
                        key
                    );
                }
            }
        }
    }

    #[test]
    fn test_22_humanize_offsets_in_range() {
        let room = MusicianRoom::new(RoomId(0), Instrument::Melody, "Lead");
        let ctx = JamContext {
            bpm: 120.0,
            key: 0,
            scale: Scale::Minor,
            bars: 4,
            energy: 0.5,
        };
        let pattern = room.generate(4, &ctx);
        for offset in &pattern.timing_offsets {
            // Humanize is roughness * 0.05 * (0..1) ≈ max ±25ms
            assert!(
                offset.abs() <= 0.06,
                "timing offset {} out of range",
                offset
            );
        }
    }
}
