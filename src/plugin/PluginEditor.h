#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "PluginProcessor.h"

// Una sola pantalla: la escala de la canción entera (y su relativa, si está en duda), su fiabilidad y un aviso si otra
// escala con notas distintas es casi igual de probable.
// Textos en inglés (como Auto-Tune y Melodyne). Con ARA el análisis empieza solo; sin ARA, «Analyze song» escucha la
// canción mientras suena; en cualquier caso se puede soltar o abrir un archivo.
class SkaleEditor : public juce::AudioProcessorEditor,
#if JucePlugin_Enable_ARA
                    public juce::AudioProcessorEditorARAExtension,
#endif
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
    void paintResult(juce::Graphics&, juce::Rectangle<int> area);
    void paintMessage(juce::Graphics&, juce::Rectangle<int> area, const juce::String& title, const juce::String& body);
    void chooseFile();

    SkaleProcessor& processor_;
    SongState state_;
    bool dragOver_ = false;
    juce::TextButton analyzeButton_{"Analyze song"};
    juce::TextButton finishButton_{"Finish now"};
    juce::TextButton openButton_{"Open file..."};
    std::unique_ptr<juce::FileChooser> chooser_;
};
