#include "PluginEditor.h"

SkaleEditor::SkaleEditor(SkaleProcessor& p) : juce::AudioProcessorEditor(&p), processor_(p) {
    setSize(420, 260);
    addAndMakeVisible(resetButton_);
    resetButton_.onClick = [this] { processor_.resetAnalysis(); };
    startTimerHz(10);
}

void SkaleEditor::refresh() {
    snap_ = processor_.latest();
    repaint();
}

void SkaleEditor::resized() { resetButton_.setBounds(getWidth() - 100, getHeight() - 36, 88, 26); }

void SkaleEditor::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff15181d));
    auto area = getLocalBounds().reduced(16);

    g.setColour(juce::Colours::white.withAlpha(0.55f));
    g.setFont(juce::FontOptions(13.f));
    g.drawText("Skale", area.removeFromTop(18), juce::Justification::left);

    if (!snap_.valid) {
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::FontOptions(18.f));
        g.drawText(snap_.listenedSeconds > 0 ? "Escuchando..." : "Esperando audio...", area,
                   juce::Justification::centred);
        return;
    }

    g.setColour(juce::Colour(0xff5bd6a6));
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

    g.setFont(juce::FontOptions(20.f));
    g.drawText("Acorde:  " + juce::String(snap_.chordName), area.removeFromTop(30), juce::Justification::left);
}
