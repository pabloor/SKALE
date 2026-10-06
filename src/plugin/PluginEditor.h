#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "PluginProcessor.h"

// Vista mínima: tonalidad, confianza, notas de la escala y acorde que suena.
class SkaleEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit SkaleEditor(SkaleProcessor&);
    ~SkaleEditor() override { stopTimer(); }

    void paint(juce::Graphics&) override;
    void resized() override;

    // Lee la instantánea del procesador y repinta (lo llama el temporizador).
    void refresh();

private:
    void timerCallback() override { refresh(); }

    SkaleProcessor& processor_;
    skale::TrackerSnapshot snap_;
    juce::TextButton resetButton_{"Reiniciar"};
};
