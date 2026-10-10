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
    double doubtRatio = 0.5;     // con las redes: si la 2ª tonalidad tiene al menos esta fracción de la probabilidad
                                 // de la 1ª, el resultado es dudoso y se dan las dos (0 = nunca)
    double endingSeconds = 4;    // ventana final que se mira para desempatar la tonalidad
    double endingMargin = 1e9;   // el final solo desempata candidatas a menos de este margen
    ChromaParams chroma;         // ajustes del cromagrama
    bool keepFrames = false;     // guarda la serie de cromagramas finos (para entrenar redes)
    double windowSeconds = 8;    // ventanas del voto por ventanas (paso = la mitad)
    bool cnn = true;             // combina la red convolucional con el modelo aprendido (cnnWeight); si learnedModel es false solo la red
    // Peso de las redes frente al modelo lineal (1 = solo las redes). El lineal mira el acorde final: ayuda en
    // piezas completas (203 piezas fuera del entrenamiento: 82,8 -> 85,7 %) y resta en fragmentos de 30 s
    // (FMA: 63,4 -> 62,3 %), así que depende de la duración.
    double cnnWeight = 1.0;          // audio de menos de longSeconds
    double cnnWeightLong = 0.85;     // audio de longSeconds o más
    double longSeconds = 60;
    bool learnedModel = true;    // modelo de tonalidad aprendido (ignora perfil y pesos de bajo/final); false = clásico
    double bassWeight = 1;       // peso del bajo (tónica y quinta) en la tonalidad; 0 = off
    double chordWeight = 0;      // peso de los acordes detectados (diatónicos y de tónica); 0 = off
    double endingWeight = 0.5;   // 0 = no usar el final (ver KeyDetector::detect)
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

struct WindowChroma {
    double start = 0;
    Chroma12 chroma{};   // suma 1
    Chroma12 bass{};     // suma 1
};

struct FrameChroma {
    Chroma36 chroma{};   // 36 bins por octava (3 por semitono, bin 3n = nota n a A=440), suma 1
    Chroma36 bass{};
    Chroma36 mid{};
    Chroma36 high{};
    LogSpec spec{};
};

struct SongAnalysis {
    bool valid = false;    // false si no había audio utilizable (silencio)
    double durationSeconds = 0;
    double tuningCents = 0;

    Chroma12 chroma{};
    Chroma12 bassChroma{};      // cromagrama del bajo (40-250 Hz), suma 1
    Chroma12 endingChroma{};    // cromagrama de los últimos endingSeconds, suma 1
    std::vector<FrameChroma> frames;     // solo con keepFrames; fotogramas de 2 en 2 promediados (~0,19 s)
    std::vector<WindowChroma> windows;   // ventanas deslizantes de 8 s (hop 4 s), para el modelo por ventanas
    Chroma12 startChroma{};     // cromagrama de los primeros endingSeconds, suma 1
    std::vector<KeyCandidate> candidates;   // las más probables, la primera es la elegida
    Key key;
    std::string keyName;
    std::vector<std::string> scaleNotes;    // 7 notas de la tonalidad elegida
    std::vector<DiatonicChord> diatonic;

    // Resultado dudoso (ver AnalysisOptions::doubtRatio): la 2ª tonalidad también es probable. En la partición
    // de prueba ocurre en ~1 de cada 4 pistas y la correcta está entre las dos mostradas el 74 % de las veces.
    bool doubtful = false;
    Key secondKey;
    std::string secondKeyName;
    bool secondSameNotes = false;            // la 2ª es la relativa: misma escala
    std::vector<std::string> secondScaleNotes;

    std::vector<TimelineEntry> timeline;
    std::vector<ChordUsage> chordUsage;     // ordenado de más a menos usado
};

// Analiza una canción completa (audio mono en memoria).
SongAnalysis analyze(const float* mono, std::size_t n, double sampleRate,
                     const AnalysisOptions& options = {});

}  // namespace skale
