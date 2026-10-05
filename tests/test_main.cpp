// Tests sin dependencias: audio sintético (acordes con armónicos) contra el
// analizador completo.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "analysis/Analyzer.h"
#include "analysis/AudioFile.h"
#include "analysis/Chromagram.h"
#include "analysis/Fft.h"
#include "analysis/Theory.h"

using namespace skale;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            ++g_failures;                                                     \
            std::printf("FALLO %s:%d  %s\n      ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                         \
            std::printf("\n");                                                \
        }                                                                     \
    } while (0)

constexpr double kRate = 44100.0;
const double kPi = std::acos(-1.0);

double midiToHz(double midi, double detuneCents = 0) {
    return 440.0 * std::pow(2.0, (midi - 69.0 + detuneCents / 100.0) / 12.0);
}

// Suma una nota con 6 armónicos (1/h) y una envolvente corta.
void addNote(std::vector<float>& out, std::size_t start, std::size_t len, double midi, float amp, double detune) {
    const double f0 = midiToHz(midi, detune);
    const std::size_t fade = std::size_t(0.02 * kRate);
    for (std::size_t i = 0; i < len && start + i < out.size(); ++i) {
        double env = 1.0;
        if (i < fade) env = double(i) / double(fade);
        if (len - i < fade) env = double(len - i) / double(fade);
        double s = 0;
        for (int h = 1; h <= 6; ++h) {
            if (f0 * h > kRate / 2.2) break;
            s += std::sin(2.0 * kPi * f0 * h * double(i) / kRate) / h;
        }
        out[start + i] += float(amp * env * s);
    }
}

struct SynthChord {
    int rootMidi;               // raíz de la tríada (se añade un bajo una octava debajo)
    std::vector<int> intervals; // sobre la raíz
};

std::vector<float> synthSong(const std::vector<SynthChord>& chords, double secondsPerChord, int loops = 1,
                             double detune = 0) {
    const std::size_t len = std::size_t(secondsPerChord * kRate);
    std::vector<float> out(len * chords.size() * std::size_t(loops), 0.f);
    std::size_t pos = 0;
    for (int l = 0; l < loops; ++l) {
        for (const auto& c : chords) {
            for (int i : c.intervals) addNote(out, pos, len, c.rootMidi + i, 0.12f, detune);
            addNote(out, pos, len, c.rootMidi - 12, 0.18f, detune);
            pos += len;
        }
    }
    return out;
}

SynthChord triad(int tonic, int semitonesAboveTonic, bool minor) {
    return {48 + tonic + semitonesAboveTonic, minor ? std::vector<int>{0, 3, 7} : std::vector<int>{0, 4, 7}};
}

// Progresiones en la tonalidad (tonic 0..11)
std::vector<SynthChord> majorProgression(int t) {
    return {triad(t, 0, false), triad(t, 9, true), triad(t, 5, false), triad(t, 7, false)};  // I vi IV V
}
std::vector<SynthChord> minorProgression(int t) {
    // i iv V i: la dominante mayor (armonía menor) fija el modo. Am F C G sería
    // indistinguible de Do mayor: mismas notas, mismos acordes.
    return {triad(t, 0, true), triad(t, 5, true), triad(t, 7, false), triad(t, 0, true)};
}

void testTheory() {
    {
        Speller s({6, Mode::Major});  // pc 6 mayor se escribe con sostenidos
        std::vector<std::string> got;
        for (auto n : s.scale()) got.push_back(formatNote(n, false));
        const std::vector<std::string> want = {"F#", "G#", "A#", "B", "C#", "D#", "E#"};
        CHECK(got == want, "F# mayor: %s %s %s %s %s %s %s", got[0].c_str(), got[1].c_str(), got[2].c_str(),
              got[3].c_str(), got[4].c_str(), got[5].c_str(), got[6].c_str());
    }
    {
        Speller s({3, Mode::Major});  // Eb mayor
        std::string joined;
        for (auto n : s.scale()) joined += formatNote(n, false) + " ";
        CHECK(joined == "Eb F G Ab Bb C D ", "Eb mayor: %s", joined.c_str());
    }
    {
        Speller s({9, Mode::Minor});
        std::string joined;
        for (auto n : s.scale()) joined += formatNote(n, false) + " ";
        CHECK(joined == "A B C D E F G ", "La menor: %s", joined.c_str());
    }
    CHECK(keyName({9, Mode::Minor}, true) == "La menor", "%s", keyName({9, Mode::Minor}, true).c_str());
    CHECK(keyName({0, Mode::Major}, false) == "C major", "%s", keyName({0, Mode::Major}, false).c_str());

    const auto chords = diatonicChords({0, Mode::Major}, false);
    const char* triads[7] = {"C", "Dm", "Em", "F", "G", "Am", "Bdim"};
    const char* sevenths[7] = {"Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5"};
    const char* romans[7] = {"I", "ii", "iii", "IV", "V", "vi", "vii°"};
    CHECK(chords.size() == 7, "7 acordes");
    for (std::size_t i = 0; i < chords.size() && i < 7; ++i) {
        CHECK(chords[i].triad == triads[i], "triada %zu: %s", i, chords[i].triad.c_str());
        CHECK(chords[i].seventh == sevenths[i], "séptima %zu: %s", i, chords[i].seventh.c_str());
        CHECK(chords[i].roman == romans[i], "romano %zu: %s", i, chords[i].roman.c_str());
    }

    const auto minor = diatonicChords({9, Mode::Minor}, false);
    const char* minorTriads[7] = {"Am", "Bdim", "C", "Dm", "Em", "F", "G"};
    for (std::size_t i = 0; i < minor.size() && i < 7; ++i) {
        CHECK(minor[i].triad == minorTriads[i], "La menor, triada %zu: %s", i, minor[i].triad.c_str());
    }
}

void testFft() {
    Fft fft(1024);
    std::vector<std::complex<float>> buf(1024);
    for (std::size_t i = 0; i < 1024; ++i) buf[i] = {float(std::sin(2.0 * kPi * 50.0 * double(i) / 1024.0)), 0.f};
    fft.forward(buf);
    std::size_t peak = 0;
    for (std::size_t i = 1; i < 512; ++i) {
        if (std::abs(buf[i]) > std::abs(buf[peak])) peak = i;
    }
    CHECK(peak == 50, "pico de la FFT en %zu", peak);
}

void testChroma() {
    // Un La 440 puro debe caer en la clase de nota La (9) y con afinación 0.
    std::vector<float> a(std::size_t(3 * kRate), 0.f);
    addNote(a, 0, a.size(), 69, 0.3f, 0);
    ChromaExtractor ex(kRate);
    auto frames = ex.process(a.data(), a.size());
    CHECK(!frames.empty(), "hay fotogramas");
    Chroma36 acc{};
    for (auto& f : frames) for (std::size_t i = 0; i < 36; ++i) acc[i] += f.chroma[i];
    CHECK(estimateTuning(acc) == 0, "afinación %d", estimateTuning(acc));
    Chroma12 c = foldChroma(acc, 0);
    normalize(c);
    int best = 0;
    for (int i = 1; i < 12; ++i) if (c[std::size_t(i)] > c[std::size_t(best)]) best = i;
    CHECK(best == 9, "clase de nota dominante: %d", best);

    // El mismo tono 30 cents alto: la afinación estimada debe ser +1.
    std::vector<float> b(std::size_t(3 * kRate), 0.f);
    addNote(b, 0, b.size(), 69, 0.3f, 30);
    ChromaExtractor ex2(kRate);
    Chroma36 acc2{};
    for (auto& f : ex2.process(b.data(), b.size())) for (std::size_t i = 0; i < 36; ++i) acc2[i] += f.chroma[i];
    CHECK(estimateTuning(acc2) == 1, "afinación estimada %d (esperado +1)", estimateTuning(acc2));

    // En streaming, el resultado no depende del tamaño de bloque.
    ChromaExtractor ex3(kRate);
    std::size_t total = 0;
    for (std::size_t pos = 0; pos < a.size(); pos += 1000) {
        total += ex3.process(a.data() + pos, std::min<std::size_t>(1000, a.size() - pos)).size();
    }
    CHECK(total == frames.size(), "bloques: %zu fotogramas vs %zu", total, frames.size());
}

void testKeys() {
    for (int tonic = 0; tonic < 12; ++tonic) {
        for (int m = 0; m < 2; ++m) {
            const bool minor = m == 1;
            const auto song = synthSong(minor ? minorProgression(tonic) : majorProgression(tonic), 2.0, 2);
            const SongAnalysis a = analyze(song.data(), song.size(), kRate);
            CHECK(a.valid, "análisis válido");
            if (!a.valid) continue;
            CHECK(a.key.tonic == tonic && a.key.mode == (minor ? Mode::Minor : Mode::Major),
                  "esperado %s, obtenido %s", keyName({tonic, minor ? Mode::Minor : Mode::Major}, false).c_str(),
                  a.keyName.c_str());
            for (const auto& u : a.chordUsage) {
                CHECK(u.diatonic, "%s no debería salir fuera de %s", u.name.c_str(), a.keyName.c_str());
            }
        }
    }
}

void testDetunedKey() {
    const auto song = synthSong(majorProgression(7), 2.0, 2, -30);  // Sol mayor, 30 cents bajo
    const SongAnalysis a = analyze(song.data(), song.size(), kRate);
    CHECK(a.valid && a.key.tonic == 7 && a.key.mode == Mode::Major, "desafinada: %s", a.keyName.c_str());
    CHECK(a.tuningCents < 0, "afinación estimada %+.0f cents", a.tuningCents);
}

void testChords() {
    // C Am F G, 2 s cada uno, dos vueltas.
    const auto song = synthSong(majorProgression(0), 2.0, 2);
    const SongAnalysis a = analyze(song.data(), song.size(), kRate);
    CHECK(a.valid, "válido");
    std::vector<std::string> seq;
    for (const auto& e : a.timeline) {
        if (e.end - e.start >= 1.0) seq.push_back(e.name);
    }
    const std::vector<std::string> want = {"C", "Am", "F", "G", "C", "Am", "F", "G"};
    CHECK(seq == want, "secuencia de acordes distinta (%zu segmentos)", seq.size());
    if (seq != want) {
        for (const auto& e : a.timeline) std::printf("      %.2f-%.2f %s\n", e.start, e.end, e.name.c_str());
    }
    for (const auto& u : a.chordUsage) CHECK(u.diatonic, "%s debería ser diatónico", u.name.c_str());

    // Acorde fuera de la tonalidad: D mayor entre acordes de Do mayor.
    // I vi IV V I IV II V I: casi todo en Do mayor, con un único D mayor prestado.
    std::vector<SynthChord> prog = {triad(0, 0, false), triad(0, 9, true), triad(0, 5, false), triad(0, 7, false),
                                    triad(0, 0, false), triad(0, 5, false), triad(0, 2, false), triad(0, 7, false),
                                    triad(0, 0, false), triad(0, 5, false), triad(0, 7, false), triad(0, 0, false)};
    const auto song2 = synthSong(prog, 2.0, 1);
    const SongAnalysis b = analyze(song2.data(), song2.size(), kRate);
    bool sawD = false;
    for (const auto& u : b.chordUsage) if (u.name == "D") { sawD = true; CHECK(!u.diatonic, "D no es diatónico en Do"); }
    CHECK(b.valid && b.key.tonic == 0 && b.key.mode == Mode::Major, "tonalidad: %s", b.keyName.c_str());
    CHECK(sawD, "se esperaba detectar el acorde D");
}

void testSilence() {
    std::vector<float> silence(std::size_t(3 * kRate), 0.f);
    const SongAnalysis a = analyze(silence.data(), silence.size(), kRate);
    CHECK(!a.valid, "el silencio no es válido");
    const SongAnalysis e = analyze(nullptr, 0, kRate);
    CHECK(!e.valid, "vacío no es válido");
}

void writeWav16(const std::string& path, const std::vector<float>& mono, int channels, int rate) {
    std::ofstream f(path, std::ios::binary);
    const std::uint32_t dataBytes = std::uint32_t(mono.size() * std::size_t(channels) * 2);
    auto w32 = [&](std::uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto w16 = [&](std::uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("RIFF", 4); w32(36 + dataBytes); f.write("WAVEfmt ", 8);
    w32(16); w16(1); w16(std::uint16_t(channels)); w32(std::uint32_t(rate));
    w32(std::uint32_t(rate * channels * 2)); w16(std::uint16_t(channels * 2)); w16(16);
    f.write("data", 4); w32(dataBytes);
    for (float s : mono) {
        const auto v = std::int16_t(std::lround(std::max(-1.f, std::min(1.f, s)) * 32767.f));
        for (int c = 0; c < channels; ++c) f.write(reinterpret_cast<const char*>(&v), 2);
    }
}

void testWavFile() {
    const auto song = synthSong(minorProgression(9), 2.0, 2);  // La menor
    const std::string path = "skale_test.wav";
    writeWav16(path, song, 2, int(kRate));

    AudioData audio;
    std::string error;
    CHECK(loadAudioFile(path, audio, error), "carga: %s", error.c_str());
    CHECK(audio.channels == 2 && audio.sampleRate == kRate, "formato");
    CHECK(audio.mono.size() == song.size(), "longitud %zu vs %zu", audio.mono.size(), song.size());
    const SongAnalysis a = analyze(audio.mono.data(), audio.mono.size(), audio.sampleRate);
    CHECK(a.valid && a.key.tonic == 9 && a.key.mode == Mode::Minor, "wav: %s", a.keyName.c_str());
    std::remove(path.c_str());

    CHECK(!loadAudioFile("no_existe.wav", audio, error), "archivo inexistente");
    CHECK(!loadAudioFile("cancion.flac", audio, error), "formato no soportado");
}

}  // namespace

int main() {
    testTheory();
    testFft();
    testChroma();
    testKeys();
    testDetunedKey();
    testChords();
    testSilence();
    testWavFile();
    std::printf("%d comprobaciones, %d fallos\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
