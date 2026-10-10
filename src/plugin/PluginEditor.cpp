#include "PluginEditor.h"

namespace {

const juce::Colour kBg(0xff15181d), kAccent(0xff5bd6a6), kWarn(0xffe6b450);

bool isAudioFile(const juce::String& path) {
    const auto ext = juce::File(path).getFileExtension().toLowerCase();
    return ext == ".wav" || ext == ".aiff" || ext == ".aif" || ext == ".flac" || ext == ".ogg" || ext == ".mp3";
}

}  // namespace

SkaleEditor::SkaleEditor(SkaleProcessor& p) : juce::AudioProcessorEditor(&p), processor_(p) {
    setSize(460, 360);
    addAndMakeVisible(resetButton_);
    addAndMakeVisible(openButton_);
    addChildComponent(backButton_);
    resetButton_.onClick = [this] { processor_.resetAnalysis(); };
    openButton_.onClick = [this] { chooseFile(); };
    backButton_.onClick = [this] { processor_.clearFileAnalysis(); refresh(); };
    startTimerHz(10);
}

void SkaleEditor::refresh() {
    snap_ = processor_.latest();
    file_ = processor_.fileState();
    const bool fileMode = file_.status != FileAnalysisState::Status::Idle;
    backButton_.setVisible(fileMode);
    resetButton_.setVisible(!fileMode);
    repaint();
}

void SkaleEditor::resized() {
    const int y = getHeight() - 36;
    openButton_.setBounds(12, y, 130, 26);
    resetButton_.setBounds(getWidth() - 100, y, 88, 26);
    backButton_.setBounds(getWidth() - 150, y, 138, 26);
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
    chooser_ = std::make_unique<juce::FileChooser>("Elige un archivo de audio", juce::File(),
                                                   "*.wav;*.aiff;*.aif;*.flac;*.ogg;*.mp3");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this](const juce::FileChooser& fc) {
                              const auto f = fc.getResult();
                              if (f.existsAsFile()) { processor_.analyzeFile(f); refresh(); }
                          });
}

void SkaleEditor::paint(juce::Graphics& g) {
    g.fillAll(kBg);
    auto area = getLocalBounds().reduced(16);
    area.removeFromBottom(36);

    g.setColour(juce::Colours::white.withAlpha(0.55f));
    g.setFont(juce::FontOptions(13.f));
    const bool fileMode = file_.status != FileAnalysisState::Status::Idle;
    g.drawText(fileMode ? "Skale  -  archivo" : "Skale  -  en directo", area.removeFromTop(18),
               juce::Justification::left);

    if (fileMode) paintFile(g, area); else paintLive(g, area);

    if (dragOver_) {
        g.setColour(kAccent.withAlpha(0.18f));
        g.fillRect(getLocalBounds());
        g.setColour(kAccent);
        g.drawRect(getLocalBounds(), 3);
        g.setFont(juce::FontOptions(22.f, juce::Font::bold));
        g.drawText("Suelta el archivo para analizarlo", getLocalBounds(), juce::Justification::centred);
    }
}

void SkaleEditor::paintLive(juce::Graphics& g, juce::Rectangle<int> area) {
    if (!snap_.valid) {
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::FontOptions(18.f));
        g.drawText(snap_.listenedSeconds > 0 ? "Escuchando..." : "Esperando audio...",
                   area.removeFromTop(area.getHeight() - 30), juce::Justification::centred);
        g.setColour(juce::Colours::white.withAlpha(0.45f));
        g.setFont(juce::FontOptions(13.f));
        g.drawText("O arrastra aqui un archivo de audio", area, juce::Justification::centred);
        return;
    }

    g.setColour(kAccent);
    g.setFont(juce::FontOptions(40.f, juce::Font::bold));
    g.drawText(snap_.keyName, area.removeFromTop(56), juce::Justification::left);

    g.setColour(juce::Colours::white.withAlpha(0.65f));
    g.setFont(juce::FontOptions(14.f));
    juce::String conf = "Confianza " + juce::String(int(snap_.confidence * 100.f + 0.5f)) + " %";
    if (snap_.candidates.size() > 1) {
        conf += "   |   alternativas: ";
        for (std::size_t i = 1; i < snap_.candidates.size() && i < 3; ++i) {
            conf += juce::String(skale::keyName(snap_.candidates[i].key, false)) + "  ";
        }
    }
    g.drawText(conf, area.removeFromTop(22), juce::Justification::left);

    area.removeFromTop(10);
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(20.f));
    juce::String scale;
    for (const auto& n : snap_.scaleNotes) scale += juce::String(n) + "   ";
    g.drawText("Escala:  " + scale, area.removeFromTop(30), juce::Justification::left);
    g.drawText("Acorde:  " + juce::String(snap_.chordName), area.removeFromTop(30), juce::Justification::left);
}

void SkaleEditor::paintFile(juce::Graphics& g, juce::Rectangle<int> area) {
    g.setColour(juce::Colours::white.withAlpha(0.8f));
    g.setFont(juce::FontOptions(14.f));
    g.drawText(file_.fileName, area.removeFromTop(20), juce::Justification::left);

    if (file_.status == FileAnalysisState::Status::Working) {
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::FontOptions(18.f));
        g.drawText("Analizando...", area, juce::Justification::centred);
        return;
    }
    if (file_.status == FileAnalysisState::Status::Error) {
        g.setColour(kWarn);
        g.setFont(juce::FontOptions(15.f));
        g.drawFittedText(file_.error, area.reduced(0, 30), juce::Justification::centred, 3);
        return;
    }

    const auto& r = file_.result;
    auto keyRow = area.removeFromTop(52);
    g.setColour(kAccent);
    const juce::Font big(juce::FontOptions(40.f, juce::Font::bold));
    g.setFont(big);
    g.drawText(r.keyName, keyRow, juce::Justification::left);
    if (r.doubtful) {   // dos tonalidades parecidas de probables: se dan las dos
        keyRow.removeFromLeft(int(juce::GlyphArrangement::getStringWidth(big, r.keyName)) + 12);
        g.setColour(kAccent.withAlpha(0.7f));
        g.setFont(juce::FontOptions(26.f, juce::Font::bold));
        g.drawText("o " + juce::String(r.secondKeyName), keyRow, juce::Justification::left);
    }

    g.setColour(juce::Colours::white.withAlpha(0.65f));
    g.setFont(juce::FontOptions(14.f));
    auto pct = [](float c) { return juce::String(int(c * 100.f + 0.5f)) + " %"; };
    juce::String info;
    if (r.doubtful) {
        info = "Dudosa: " + pct(r.candidates[0].confidence) + " / " + pct(r.candidates[1].confidence);
        if (r.secondSameNotes) info += "  (mismas notas)";
    } else {
        info = "Confianza " + pct(r.candidates.empty() ? 0.f : r.candidates[0].confidence);
        if (r.candidates.size() > 1) {
            info += "   |   alternativas: ";
            for (std::size_t i = 1; i < r.candidates.size() && i < 3; ++i) info += juce::String(skale::keyName(r.candidates[i].key, false)) + "  ";
        }
    }
    info += juce::String::formatted("   |   %d:%02d", int(r.durationSeconds) / 60, int(r.durationSeconds) % 60);
    g.drawText(info, area.removeFromTop(20), juce::Justification::left);

    area.removeFromTop(8);
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(17.f));
    juce::String scale;
    for (const auto& n : r.scaleNotes) scale += juce::String(n) + "  ";
    g.drawText("Escala:  " + scale, area.removeFromTop(26), juce::Justification::left);
    if (r.doubtful && !r.secondSameNotes) {
        juce::String scale2;
        for (const auto& n : r.secondScaleNotes) scale2 += juce::String(n) + "  ";
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::FontOptions(14.f));
        g.drawText("o, si es " + juce::String(r.secondKeyName) + ":  " + scale2, area.removeFromTop(20), juce::Justification::left);
        area.removeFromTop(6);
    }

    juce::String diat;
    for (const auto& d : r.diatonic) diat += juce::String(d.roman) + " " + juce::String(d.triad) + "    ";
    g.setFont(juce::FontOptions(14.f));
    g.setColour(juce::Colours::white.withAlpha(0.85f));
    g.drawFittedText("Acordes de la tonalidad:  " + diat, area.removeFromTop(40), juce::Justification::topLeft, 2);

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(14.f));
    juce::String used;
    int count = 0;
    for (const auto& u : r.chordUsage) {
        if (count++ >= 6) break;
        used += juce::String(u.name) + (u.diatonic ? "" : "*") + " " + juce::String(int(u.fraction * 100.0 + 0.5)) + " %    ";
    }
    g.drawFittedText("Mas usados:  " + used, area.removeFromTop(40), juce::Justification::topLeft, 2);
    g.setColour(juce::Colours::white.withAlpha(0.45f));
    g.setFont(juce::FontOptions(12.f));
    g.drawText("* = acorde ajeno a la tonalidad", area.removeFromTop(18), juce::Justification::left);
}
