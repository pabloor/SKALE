#include "PluginEditor.h"

#include "analysis/Theory.h"

namespace {

const juce::Colour kBg(0xff15181d), kAccent(0xff5bd6a6), kWarn(0xffe6b450);

bool isAudioFile(const juce::String& path) {
    const auto ext = juce::File(path).getFileExtension().toLowerCase();
    return ext == ".wav" || ext == ".aiff" || ext == ".aif" || ext == ".flac" || ext == ".ogg" || ext == ".mp3";
}

skale::Key relativeOf(const skale::Key& k) {
    return k.mode == skale::Mode::Major ? skale::Key{(k.tonic + 9) % 12, skale::Mode::Minor}
                                        : skale::Key{(k.tonic + 3) % 12, skale::Mode::Major};
}
bool sameKey(const skale::Key& a, const skale::Key& b) { return a.tonic == b.tonic && a.mode == b.mode; }

// La relativa de la 1ª si su probabilidad es al menos kRelativeRatio veces la de la 1ª (con canciones completas
// de FMA aparece en ~1 de cada 5 y la correcta está entre las dos el 74,8 % frente al 69,8 % de la 1ª sola).
constexpr float kRelativeRatio = 0.3f;

const skale::KeyCandidate* relativeInDoubt(const skale::SongAnalysis& r) {
    if (r.candidates.size() < 2) return nullptr;
    const skale::Key rel = relativeOf(r.candidates[0].key);
    for (std::size_t i = 1; i < r.candidates.size(); ++i)
        if (r.candidates[i].key.tonic == rel.tonic && r.candidates[i].key.mode == rel.mode)
            return r.candidates[i].confidence >= kRelativeRatio * r.candidates[0].confidence ? &r.candidates[i] : nullptr;
    return nullptr;
}

// Probabilidad de las 7 notas de la 1ª: la suya más la de su relativa.
float notesProbability(const skale::SongAnalysis& r) {
    if (r.candidates.empty()) return 0.f;
    float p = r.candidates[0].confidence;
    const auto rel = relativeOf(r.candidates[0].key);
    for (std::size_t i = 1; i < r.candidates.size(); ++i)
        if (sameKey(r.candidates[i].key, rel)) p += r.candidates[i].confidence;
    return p;
}

// La escala más probable con otras notas (ni la 1ª ni su relativa) si tiene al menos la mitad de probabilidad que la
// 1ª: con canciones completas sale en ~1 de cada 5 y las notas correctas están entre las mostradas el 83 % de las
// veces (FMA; GTZAN 82 %), frente al 78 / 74 % sin aviso.
const skale::KeyCandidate* otherScaleInDoubt(const skale::SongAnalysis& r) {
    if (r.candidates.size() < 2) return nullptr;
    const auto rel = relativeOf(r.candidates[0].key);
    for (std::size_t i = 1; i < r.candidates.size(); ++i) {
        if (sameKey(r.candidates[i].key, rel)) continue;
        return r.candidates[i].confidence >= 0.5f * r.candidates[0].confidence ? &r.candidates[i] : nullptr;
    }
    return nullptr;
}

juce::String clock(double s) { return juce::String::formatted("%d:%02d", int(s) / 60, int(s) % 60); }

}  // namespace

SkaleEditor::SkaleEditor(SkaleProcessor& p)
    : juce::AudioProcessorEditor(&p),
#if JucePlugin_Enable_ARA
      juce::AudioProcessorEditorARAExtension(&p),
#endif
      processor_(p) {
    setSize(480, 290);
    addAndMakeVisible(analyzeButton_);
    addAndMakeVisible(openButton_);
    addChildComponent(finishButton_);
    analyzeButton_.onClick = [this] { processor_.startListening(); refresh(); };
    finishButton_.onClick = [this] { processor_.stopListening(); refresh(); };
    openButton_.onClick = [this] { chooseFile(); };
    startTimerHz(5);
    refresh();
}

void SkaleEditor::refresh() {
    state_ = processor_.state();
    const bool listening = state_.status == SongState::Status::Listening;
    finishButton_.setVisible(listening);
    analyzeButton_.setVisible(!listening && !processor_.usesARA());
    analyzeButton_.setButtonText(state_.status == SongState::Status::Done ? "Analyze again" : "Analyze song");
    repaint();
}

void SkaleEditor::resized() {
    const int y = getHeight() - 38;
    analyzeButton_.setBounds(14, y, 140, 26);
    finishButton_.setBounds(14, y, 140, 26);
    openButton_.setBounds(getWidth() - 134, y, 120, 26);
}

bool SkaleEditor::isInterestedInFileDrag(const juce::StringArray& files) {
    for (const auto& f : files) if (isAudioFile(f)) return true;
    return false;
}

void SkaleEditor::filesDropped(const juce::StringArray& files, int, int) {
    dragOver_ = false;
    for (const auto& f : files) {
        if (isAudioFile(f)) { processor_.analyzeFile(juce::File(f)); break; }
    }
    refresh();
}

void SkaleEditor::chooseFile() {
    chooser_ = std::make_unique<juce::FileChooser>("Choose an audio file", juce::File(), "*.wav;*.aiff;*.aif;*.flac;*.ogg;*.mp3");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this](const juce::FileChooser& fc) {
                              const auto f = fc.getResult();
                              if (f.existsAsFile()) { processor_.analyzeFile(f); refresh(); }
                          });
}

void SkaleEditor::paintMessage(juce::Graphics& g, juce::Rectangle<int> area, const juce::String& title, const juce::String& body) {
    area.removeFromTop(30);
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(22.f, juce::Font::bold));
    g.drawText(title, area.removeFromTop(34), juce::Justification::centred);
    g.setColour(juce::Colours::white.withAlpha(0.7f));
    g.setFont(juce::FontOptions(14.f));
    g.drawFittedText(body, area.removeFromTop(120).reduced(20, 0), juce::Justification::centredTop, 6);
}

void SkaleEditor::paint(juce::Graphics& g) {
    g.fillAll(kBg);
    auto area = getLocalBounds().reduced(16);
    area.removeFromBottom(40);

    g.setColour(juce::Colours::white.withAlpha(0.55f));
    g.setFont(juce::FontOptions(13.f));
    g.drawText("SKALE  -  song key & scale", area.removeFromTop(18), juce::Justification::left);

    using S = SongState::Status;
    switch (state_.status) {
        case S::Idle:
            if (processor_.usesARA())
                paintMessage(g, area, "Waiting for audio", "Put the instrumental (or the full mix) on this track.\n"
                                                           "Skale reads the whole region and finds its scale - no need to play it.");
            else
                paintMessage(g, area, "Find the scale of your song",
                             "Insert Skale on the instrumental (or the mix), press Analyze song and play the whole song.\n"
                             "When playback stops you get one scale for the whole song.\n\nOr drop an audio file here.");
            break;
        case S::Listening:
            paintMessage(g, area, "Listening...  " + clock(state_.listenedSeconds),
                         "Play the song from start to end. The result appears when playback stops\n"
                         "(or press Finish now). Nothing changes while it plays.");
            break;
        case S::Working:
            paintMessage(g, area, "Analyzing...", state_.source);
            break;
        case S::Error:
            paintMessage(g, area, "Could not analyze", state_.error);
            break;
        case S::Done:
            paintResult(g, area);
            break;
    }

    if (dragOver_) {
        g.setColour(kAccent.withAlpha(0.18f));
        g.fillRect(getLocalBounds());
        g.setColour(kAccent);
        g.drawRect(getLocalBounds(), 3);
        g.setFont(juce::FontOptions(22.f, juce::Font::bold));
        g.drawText("Drop to find the scale", getLocalBounds(), juce::Justification::centred);
    }
}

void SkaleEditor::paintResult(juce::Graphics& g, juce::Rectangle<int> area) {
    const auto& r = state_.result;
    g.setColour(juce::Colours::white.withAlpha(0.5f));
    g.setFont(juce::FontOptions(12.f));
    g.drawText(state_.source + "  -  " + clock(r.durationSeconds), area.removeFromTop(18), juce::Justification::left);

    // Escala (y la relativa si está en duda: mismas notas)
    const auto* rel = relativeInDoubt(r);
    auto keyRow = area.removeFromTop(48);
    const juce::Font big(juce::FontOptions(38.f, juce::Font::bold));
    g.setColour(kAccent);
    g.setFont(big);
    g.drawText(r.keyName, keyRow, juce::Justification::left);
    if (rel != nullptr) {
        keyRow.removeFromLeft(int(juce::GlyphArrangement::getStringWidth(big, r.keyName)) + 10);
        g.setFont(juce::FontOptions(24.f, juce::Font::bold));
        g.drawText("/  " + juce::String(skale::keyName(rel->key, false)) + "   (same notes)", keyRow, juce::Justification::left);
    }
    juce::String notes;
    for (const auto& n : r.scaleNotes) notes += juce::String(n) + "   ";
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(20.f));
    g.drawText(notes.trimEnd(), area.removeFromTop(28), juce::Justification::left);

    // Fiabilidad: probabilidad de estas 7 notas (la escala más su relativa). Calibrado con canciones completas de FMA
    // y GTZAN: >= 0,75 -> ~92 % de notas correctas, 0,5-0,75 -> ~74 %, menos -> ~52 %.
    const float pNotes = notesProbability(r);
    const juce::String level = pNotes >= 0.75f ? "High" : pNotes >= 0.5f ? "Medium" : "Low";
    const juce::String hint = pNotes >= 0.75f ? "about 9 in 10 results like this are right"
                            : pNotes >= 0.5f  ? "about 3 in 4 results like this are right"
                                              : "about 1 in 2 - check the other option below by ear";
    auto relRow = area.removeFromTop(22);
    g.setColour(pNotes >= 0.75f ? kAccent : pNotes >= 0.5f ? juce::Colours::white : kWarn);
    g.setFont(juce::FontOptions(14.f, juce::Font::bold));
    g.drawText("Reliability: " + level, relRow.removeFromLeft(140), juce::Justification::left);
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.setFont(juce::FontOptions(13.f));
    g.drawText("(" + hint + ")", relRow, juce::Justification::left);

    // Aviso: otra escala con notas distintas casi igual de probable.
    if (const auto* other = otherScaleInDoubt(r)) {
        area.removeFromTop(8);
        const skale::Speller a(r.key), b(other->key);
        juce::StringArray out, in;
        for (int pc : a.scalePitchClasses()) if (!b.inScale(pc)) out.add(a.name(pc, false));
        for (int pc : b.scalePitchClasses()) if (!a.inScale(pc)) in.add(b.name(pc, false));
        g.setColour(kWarn);
        g.setFont(juce::FontOptions(14.f, juce::Font::bold));
        g.drawFittedText("Not sure: it could also be " + juce::String(skale::keyName(other->key, false)) + " ("
                             + juce::String(juce::roundToInt(other->confidence * 100.f)) + " %), which changes "
                             + out.joinIntoString(" ") + "  ->  " + in.joinIntoString(" ") + ". Compare both by ear.",
                         area.removeFromTop(40), juce::Justification::topLeft, 2);
    }
}
