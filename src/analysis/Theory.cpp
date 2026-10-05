#include "analysis/Theory.h"

#include <algorithm>

namespace skale {

namespace {

constexpr int kNaturalPc[7] = {0, 2, 4, 5, 7, 9, 11};
constexpr int kMajorScale[7] = {0, 2, 4, 5, 7, 9, 11};
constexpr int kMinorScale[7] = {0, 2, 3, 5, 7, 8, 10};

// Escritura de la tónica: la más habitual para cada tonalidad.
constexpr Note kMajorTonic[12] = {{0, 0}, {1, -1}, {1, 0}, {2, -1}, {2, 0}, {3, 0},
                                  {3, 1}, {4, 0}, {5, -1}, {5, 0}, {6, -1}, {6, 0}};
constexpr Note kMinorTonic[12] = {{0, 0}, {0, 1}, {1, 0}, {2, -1}, {2, 0}, {3, 0},
                                  {3, 1}, {4, 0}, {4, 1}, {5, 0}, {6, -1}, {6, 0}};

constexpr Note kSharpNames[12] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {3, 0},
                                  {3, 1}, {4, 0}, {4, 1}, {5, 0}, {5, 1}, {6, 0}};
constexpr Note kFlatNames[12] = {{0, 0}, {1, -1}, {1, 0}, {2, -1}, {2, 0}, {3, 0},
                                 {4, -1}, {4, 0}, {5, -1}, {5, 0}, {6, -1}, {6, 0}};

int mod12(int v) { return ((v % 12) + 12) % 12; }

const char* const kRoman[7] = {"I", "II", "III", "IV", "V", "VI", "VII"};

}  // namespace

const char* chordSuffix(ChordType t) {
    switch (t) {
        case ChordType::Major: return "";
        case ChordType::Minor: return "m";
        case ChordType::Dominant7: return "7";
        case ChordType::Major7: return "maj7";
        case ChordType::Minor7: return "m7";
        case ChordType::Diminished: return "dim";
        case ChordType::Sus2: return "sus2";
        case ChordType::Sus4: return "sus4";
    }
    return "";
}

std::vector<int> chordIntervals(ChordType t) {
    switch (t) {
        case ChordType::Major: return {0, 4, 7};
        case ChordType::Minor: return {0, 3, 7};
        case ChordType::Dominant7: return {0, 4, 7, 10};
        case ChordType::Major7: return {0, 4, 7, 11};
        case ChordType::Minor7: return {0, 3, 7, 10};
        case ChordType::Diminished: return {0, 3, 6};
        case ChordType::Sus2: return {0, 2, 7};
        case ChordType::Sus4: return {0, 5, 7};
    }
    return {};
}

std::vector<int> chordPitchClasses(int root, ChordType type) {
    std::vector<int> pcs;
    for (int i : chordIntervals(type)) pcs.push_back(mod12(root + i));
    return pcs;
}

std::string formatNote(Note n, bool solfege) {
    static const char* const letters[7] = {"C", "D", "E", "F", "G", "A", "B"};
    static const char* const solfeges[7] = {"Do", "Re", "Mi", "Fa", "Sol", "La", "Si"};
    std::string s = solfege ? solfeges[n.letter] : letters[n.letter];
    for (int i = 0; i < n.accidental; ++i) s += '#';
    for (int i = 0; i > n.accidental; --i) s += 'b';
    return s;
}

std::string keyName(Key k, bool solfege) {
    Speller sp(k);
    std::string tonic = formatNote(sp.scale()[0], solfege);
    if (solfege) return tonic + (k.mode == Mode::Major ? " mayor" : " menor");
    return tonic + (k.mode == Mode::Major ? " major" : " minor");
}

Speller::Speller(Key key) : key_(key) {
    const int tonic = mod12(key.tonic);
    const Note tn = key.mode == Mode::Major ? kMajorTonic[tonic] : kMinorTonic[tonic];
    const int* intervals = key.mode == Mode::Major ? kMajorScale : kMinorScale;

    for (int i = 0; i < 7; ++i) {
        Note n;
        n.letter = (tn.letter + i) % 7;
        const int pc = mod12(tonic + intervals[i]);
        n.accidental = mod12(pc - kNaturalPc[n.letter] + 6) - 6;
        scale_.push_back(n);
        scalePcs_[std::size_t(i)] = pc;
        if (n.accidental < 0) preferFlats_ = true;
    }
}

bool Speller::inScale(int pc) const {
    return std::find(scalePcs_.begin(), scalePcs_.end(), mod12(pc)) != scalePcs_.end();
}

Note Speller::note(int pc) const {
    pc = mod12(pc);
    for (int i = 0; i < 7; ++i) {
        if (scalePcs_[std::size_t(i)] == pc) return scale_[std::size_t(i)];
    }
    return preferFlats_ ? kFlatNames[pc] : kSharpNames[pc];
}

std::string Speller::name(int pc, bool solfege) const { return formatNote(note(pc), solfege); }

std::string chordName(int root, ChordType type, const Speller& s, bool solfege) {
    return s.name(root, solfege) + chordSuffix(type);
}

std::vector<DiatonicChord> diatonicChords(Key key, bool solfege) {
    Speller sp(key);
    const auto& pcs = sp.scalePitchClasses();
    std::vector<DiatonicChord> out;

    for (int d = 0; d < 7; ++d) {
        const int root = pcs[std::size_t(d)];
        const int third = mod12(pcs[std::size_t((d + 2) % 7)] - root);
        const int fifth = mod12(pcs[std::size_t((d + 4) % 7)] - root);
        const int sev = mod12(pcs[std::size_t((d + 6) % 7)] - root);

        const std::string rootName = sp.name(root, solfege);
        std::string roman = kRoman[d];
        std::string triadSuffix, seventhSuffix;

        if (third == 4 && fifth == 7) {
            triadSuffix = "";
            seventhSuffix = sev == 11 ? "maj7" : "7";
        } else if (third == 3 && fifth == 7) {
            triadSuffix = "m";
            seventhSuffix = sev == 11 ? "mMaj7" : "m7";
            std::transform(roman.begin(), roman.end(), roman.begin(), ::tolower);
        } else if (third == 3 && fifth == 6) {
            triadSuffix = "dim";
            seventhSuffix = sev == 9 ? "dim7" : "m7b5";
            std::transform(roman.begin(), roman.end(), roman.begin(), ::tolower);
            roman += "°";
        } else {
            triadSuffix = "aug";
            seventhSuffix = "maj7#5";
            roman += "+";
        }

        DiatonicChord c;
        c.degree = d + 1;
        c.rootPc = root;
        c.roman = roman;
        c.triad = rootName + triadSuffix;
        c.seventh = rootName + seventhSuffix;
        c.pitchClasses = {root, pcs[std::size_t((d + 2) % 7)], pcs[std::size_t((d + 4) % 7)]};
        out.push_back(c);
    }
    return out;
}

}  // namespace skale
