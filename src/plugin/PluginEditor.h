#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "PluginProcessor.h"

// Vista en directo (tonalidad, confianza, escala, acorde) y, al arrastrar o
// abrir un archivo, resultado del análisis completo de ese archivo.
class SkaleEditor : public juce::AudioProcessorEditor,
                    public juce::FileDragAndDropTarget,
                    private juce::Timer {
public:
    explicit SkaleEditor(SkaleProcessor&);
    ~SkaleEditor() override { stopTimer(); }

    void paint(juce::Graphics&) override;
    void resized() override;

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter(const juce::StringArray&, int, int) override { dragOver_ = true; repaint(); }
    void fileDragExit(const juce::StringArray&) override { dragOver_ = false; repaint(); }

    // Lee el estado del procesador y repinta (lo llama el temporizador).
    void refresh();

private:
    void timerCallback() override { refresh(); }
    void paintLive(juce::Graphics&, juce::Rectangle<int> area);
    void paintFile(juce::Graphics&, juce::Rectangle<int> area);
    void chooseFile();

    SkaleProcessor& processor_;
    skale::TrackerSnapshot snap_;
    FileAnalysisState file_;
    bool dragOver_ = false;
    juce::TextButton resetButton_{"Reiniciar"};
    juce::TextButton openButton_{"Abrir archivo..."};
    juce::TextButton backButton_{"Volver al directo"};
    std::unique_ptr<juce::FileChooser> chooser_;
};
