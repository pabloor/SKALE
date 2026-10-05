#pragma once

#include <array>
#include <string>
#include <vector>

namespace skale {

enum class Mode { Major, Minor };

struct Key {
    int tonic = 0;  // 0 = Do ... 11 = Si
    Mode mode = Mode::Major;
};

enum class ChordType { Major, Minor, Dominant7, Major7, Minor7, Diminished, Sus2, Sus4 };
constexpr int kChordTypeCount = 8;

const char* chordSuffix(ChordType t);
std::vector<int> chordIntervals(ChordType t);

// Nota escrita: letra (0 = C ... 6 = B) y alteración (-2..2).
struct Note {
    int letter = 0;
    int accidental = 0;
};

// "C#", "Bb"... o, con solfege, "Do#", "Sib"...
std::string formatNote(Note n, bool solfege);

// "A minor" / "La menor".
std::string keyName(Key k, bool solfege);

// Escribe las notas de una tonalidad con la ortografía correcta (Gb mayor tiene
// Cb, no B) y da nombre a cualquier clase de nota según el contexto: sostenidos
// o bemoles según la armadura.
class Speller {
public:
    explicit Speller(Key key);

    const std::vector<Note>& scale() const { return scale_; }
    const std::array<int, 7>& scalePitchClasses() const { return scalePcs_; }
    bool inScale(int pc) const;
    Note note(int pc) const;
    std::string name(int pc, bool solfege) const;

private:
    Key key_;
    std::vector<Note> scale_;
    std::array<int, 7> scalePcs_{};
    bool preferFlats_ = false;
};

std::string chordName(int root, ChordType type, const Speller& s, bool solfege);

struct DiatonicChord {
    int degree = 0;        // 1..7
    int rootPc = 0;
    std::string roman;     // "I", "ii", "vii°"...
    std::string triad;     // "C", "Dm", "Bdim"
    std::string seventh;   // "Cmaj7", "Dm7", "G7", "Bm7b5"
    std::vector<int> pitchClasses;  // de la triada
};

std::vector<DiatonicChord> diatonicChords(Key key, bool solfege);

// Clases de nota (0..11) que contiene un acorde.
std::vector<int> chordPitchClasses(int root, ChordType type);

}  // namespace skale
