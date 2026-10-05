#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "analysis/ChordDetector.h"
#include "analysis/Chromagram.h"
#include "analysis/KeyDetector.h"
#include "analysis/Theory.h"

namespace skale {

struct AnalysisOptions {
    KeyProfile profile = KeyProfile::Temperley;
    bool solfege = false;        // Do Re Mi en lugar de C D E
    std::size_t maxCandidates = 5;
    double endingSeconds = 4;    // ventana final que se mira para desempatar la tonalidad
    double endingWeight = 0;     // 0 = no usar el final (ver KeyDetector::detect)
};

struct ChordUsage {
    std::string name;
    ChordLabel chord;
    double seconds = 0;
    double fraction = 0;   // sobre el total de tiempo con acorde
    bool diatonic = false;
};

struct TimelineEntry {
    double start = 0;
    double end = 0;
    std::string name;      // "-" si no hay acorde
    ChordLabel chord;
    bool diatonic = false;
};

struct SongAnalysis {
    bool valid = false;    // false si no había audio utilizable (silencio)
    double durationSeconds = 0;
    double tuningCents = 0;

    Chroma12 chroma{};
    std::vector<KeyCandidate> candidates;   // las más probables, la primera es la elegida
    Key key;
    std::string keyName;
    std::vector<std::string> scaleNotes;    // 7 notas de la tonalidad elegida
    std::vector<DiatonicChord> diatonic;

    std::vector<TimelineEntry> timeline;
    std::vector<ChordUsage> chordUsage;     // ordenado de más a menos usado
};

// Analiza una canción completa (audio mono en memoria).
SongAnalysis analyze(const float* mono, std::size_t n, double sampleRate,
                     const AnalysisOptions& options = {});

}  // namespace skale
