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
    setSize(340, 190);
    addAndMakeVisible(analyzeButton_);
    addAndMakeVisible(openButton_);
    addChildComponent(finishButton_);
    analyzeButton_.onClick = [this] { processor_.analyzeAgain(); refresh(); };
    finishButton_.onClick = [this] { processor_.stopListening(); refresh(); };
    openButton_.onClick = [this] { chooseFile(); };
    startTimerHz(5);
    refresh();
}

void SkaleEditor::refresh() {
    state_ = processor_.state();
    const bool listening = state_.status == SongState::Status::Listening;
    finishButton_.setVisible(listening);
    analyzeButton_.setVisible(!listening);
    repaint();
}

void SkaleEditor::resized() {
    const int y = getHeight() - 40, w = (getWidth() - 36) / 2;
    analyzeButton_.setBounds(12, y, w, 28);
    finishButton_.setBounds(12, y, w, 28);
    openButton_.setBounds(24 + w, y, w, 28);
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

void SkaleEditor::paintMessage(juce::Graphics& g, juce::Rectangle<int> area, const juce::String& title, const juce::String&) {
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.setFont(juce::FontOptions(18.f));
    g.drawText(title, area, juce::Justification::centred);
}

void SkaleEditor::paint(juce::Graphics& g) {
    g.fillAll(kBg);
    auto area = getLocalBounds().reduced(12);
    area.removeFromBottom(40);

    using S = SongState::Status;
    switch (state_.status) {
        case S::Idle:      paintMessage(g, area, processor_.usesARA() ? "Waiting for audio..." : "Analyze or open a file", {}); break;
        case S::Listening: paintMessage(g, area, "Listening...  " + clock(state_.listenedSeconds), {}); break;
        case S::Working:   paintMessage(g, area, "Analyzing...", {}); break;
        case S::Error:     paintMessage(g, area, state_.error, {}); break;
        case S::Done:      paintResult(g, area); break;
    }

    if (dragOver_) {
        g.setColour(kAccent.withAlpha(0.18f));
        g.fillRect(getLocalBounds());
        g.setColour(kAccent);
        g.drawRect(getLocalBounds(), 3);
    }
}

// La escala en grande y, si hay una segunda en duda (la relativa o una escala con otras notas), debajo «or X».
void SkaleEditor::paintResult(juce::Graphics& g, juce::Rectangle<int> area) {
    const auto& r = state_.result;
    const skale::KeyCandidate* second = otherScaleInDoubt(r);
    if (second == nullptr) second = relativeInDoubt(r);
    auto top = second ? area.removeFromTop(area.getHeight() * 3 / 5) : area;
    g.setColour(kAccent);
    g.setFont(juce::FontOptions(42.f, juce::Font::bold));
    g.drawText(r.keyName, top, second ? juce::Justification::centredBottom : juce::Justification::centred);
    if (second) {
        g.setColour(kAccent.withAlpha(0.7f));
        g.setFont(juce::FontOptions(24.f, juce::Font::bold));
        g.drawText("or " + juce::String(skale::keyName(second->key, false)), area, juce::Justification::centredTop);
    }
}
