#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <juce_audio_formats/juce_audio_formats.h>

#include <map>

namespace {

// Mezcla a mono un trozo de un lector y lo añade a `mono`.
bool appendMono(juce::AudioFormatReader& reader, juce::int64 start, juce::int64 end, std::vector<float>& mono,
                const std::function<bool()>& cancelled, const std::function<void(juce::int64)>& advanced) {
    const int ch = int(std::max<unsigned>(1u, reader.numChannels));
    juce::AudioBuffer<float> buf(ch, 65536);
    for (juce::int64 pos = start; pos < end; pos += 65536) {
        if (cancelled()) return false;
        const int n = int(std::min<juce::int64>(65536, end - pos));
        if (!reader.read(&buf, 0, n, pos, true, true)) return false;
        for (int i = 0; i < n; ++i) {
            float sum = 0.f;
            for (int c = 0; c < ch; ++c) sum += buf.getReadPointer(c)[i];
            mono.push_back(sum / float(ch));
        }
        advanced(n);
    }
    return true;
}

}  // namespace

// Lee el audio (archivo, regiones ARA o lo ya capturado) y lo analiza entero, fuera del hilo de audio y del de la
// interfaz.
class SongAnalysisJob : public juce::ThreadPoolJob {
public:
    struct Part { std::unique_ptr<juce::AudioFormatReader> reader; juce::int64 start = 0, end = 0; };

    SongAnalysisJob(SkaleProcessor& p, int id, juce::String source)
        : juce::ThreadPoolJob("Skale"), proc_(p), id_(id), source_(std::move(source)) {}

    juce::File file;                 // o bien un archivo...
    std::vector<Part> parts;         // ...o trozos de lectores (regiones ARA)...
    std::vector<float> samples;      // ...o audio ya en mono
    double sampleRate = 44100;

    JobStatus runJob() override {
        auto cancelled = [this] { return shouldExit() || proc_.jobId_.load() != id_; };
        auto setProgress = [this](float p) { if (proc_.jobId_.load() == id_) proc_.progress_.store(p); };
        juce::int64 total = 0, done = 0;   // muestras por leer: la lectura es el primer 10 % de la barra
        auto advanced = [&](juce::int64 n) { done += n; if (total > 0) setProgress(0.1f * float(done) / float(total)); };
        setProgress(0.f);
        std::vector<float> mono;
        if (file != juce::File()) {
            juce::AudioFormatManager fm;
            fm.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file));
            if (!reader || reader->lengthInSamples <= 0) return fail("Could not read the file (wav, aiff, flac, ogg, mp3, m4a, aac, caf).");
            sampleRate = reader->sampleRate;
            mono.reserve(std::size_t(reader->lengthInSamples));
            total = reader->lengthInSamples;
            if (!appendMono(*reader, 0, reader->lengthInSamples, mono, cancelled, advanced)) return jobHasFinished;
        } else if (!parts.empty()) {
            for (auto& part : parts) total += part.end - part.start;
            for (auto& part : parts) {
                if (!appendMono(*part.reader, part.start, part.end, mono, cancelled, advanced)) {
                    if (cancelled()) return jobHasFinished;
                    return fail("Could not read the region audio from the host.");
                }
            }
        } else {
            mono = std::move(samples);
        }
        if (cancelled()) return jobHasFinished;
        if (mono.size() < std::size_t(2 * sampleRate)) return fail("Not enough audio (less than 2 seconds).");

        SongState st;
        st.source = source_;
        skale::AnalysisOptions options;
        options.progress = [&](float p) { setProgress(0.1f + 0.9f * p); };
        st.result = skale::analyze(mono.data(), mono.size(), sampleRate, options);
        if (!st.result.valid) return fail("No usable audio (silence).");
        st.status = SongState::Status::Done;
        return finish(std::move(st));
    }

private:
    JobStatus fail(const juce::String& msg) {
        SongState st;
        st.status = SongState::Status::Error;
        st.source = source_;
        st.error = msg;
        return finish(std::move(st));
    }

    JobStatus finish(SongState st) {
        std::lock_guard<std::mutex> lock(proc_.stateMutex_);
        if (proc_.jobId_.load() == id_) proc_.state_ = std::move(st);
        return jobHasFinished;
    }

    SkaleProcessor& proc_;
    int id_;
    juce::String source_;
};

#if JucePlugin_Enable_ARA
//==============================================================================
// ARA: el plugin tiene que reproducir él mismo el audio de las regiones (aquí, tal cual, sin modificar).
namespace {

struct SharedReadThread final : juce::TimeSliceThread {
    SharedReadThread() : juce::TimeSliceThread("Skale ARA read") { startThread(juce::Thread::Priority::high); }
};

class PassThroughRenderer final : public juce::ARAPlaybackRenderer {
public:
    using juce::ARAPlaybackRenderer::ARAPlaybackRenderer;

    void prepareToPlay(double rate, int maxBlock, int channels, juce::AudioProcessor::ProcessingPrecision,
                       AlwaysNonRealtime alwaysNonRealtime) override {
        sampleRate_ = rate;
        channels_ = channels;
        temp_.setSize(std::max(2, channels), maxBlock);
        readers_.clear();
        for (auto* region : getPlaybackRegions()) {
            auto* source = region->getAudioModification()->getAudioSource();
            if (readers_.count(source)) continue;
            auto reader = std::make_unique<juce::ARAAudioSourceReader>(source);
            if (alwaysNonRealtime == AlwaysNonRealtime::yes) {
                readers_[source] = std::move(reader);
            } else {
                auto buffered = std::make_unique<juce::BufferingAudioReader>(
                    reader.release(), *thread_, std::max(4 * maxBlock, juce::roundToInt(2.0 * rate)));
                buffered->setReadTimeout(0);
                readers_[source] = std::move(buffered);
            }
        }
    }

    void releaseResources() override { readers_.clear(); }

    bool processBlock(juce::AudioBuffer<float>& buffer, juce::AudioProcessor::Realtime,
                      const juce::AudioPlayHead::PositionInfo& pos) noexcept override {
        const int n = buffer.getNumSamples();
        buffer.clear();
        if (!pos.getIsPlaying()) return true;
        const auto block = juce::Range<juce::int64>::withStartAndLength(pos.getTimeInSamples().orFallback(0), n);
        bool ok = true;
        for (auto* region : getPlaybackRegions()) {
            const auto playRange = region->getSampleRange(sampleRate_, juce::ARAPlaybackRegion::IncludeHeadAndTail::no);
            auto render = block.getIntersectionWith(playRange);
            if (render.isEmpty()) continue;
            const juce::Range<juce::int64> modRange{region->getStartInAudioModificationSamples(), region->getEndInAudioModificationSamples()};
            const auto offset = modRange.getStart() - playRange.getStart();
            render = render.getIntersectionWith(modRange.movedToStartAt(playRange.getStart()));
            if (render.isEmpty()) continue;
            auto* source = region->getAudioModification()->getAudioSource();
            const auto it = readers_.find(source);
            if (it == readers_.end() || std::abs(source->getSampleRate() - sampleRate_) > 0.5) { ok = false; continue; }

            const int count = int(render.getLength()), at = int(render.getStart() - block.getStart());
            const int srcCh = std::max(1, int(source->getChannelCount()));
            temp_.setSize(std::max(srcCh, temp_.getNumChannels()), temp_.getNumSamples(), true, false, true);
            if (!it->second->read(&temp_, 0, count, render.getStart() + offset, true, true)) { ok = false; continue; }
            for (int c = 0; c < buffer.getNumChannels(); ++c)   // mono a todas las salidas; si no, canal a canal
                buffer.addFrom(c, at, temp_, srcCh == 1 ? 0 : std::min(c, srcCh - 1), 0, count);
        }
        return ok;
    }

    using juce::ARAPlaybackRenderer::processBlock;

private:
    double sampleRate_ = 44100;
    int channels_ = 2;
    juce::AudioBuffer<float> temp_;
    juce::SharedResourcePointer<SharedReadThread> thread_;
    std::map<juce::ARAAudioSource*, std::unique_ptr<juce::AudioFormatReader>> readers_;
};

class SkaleDocumentController final : public juce::ARADocumentControllerSpecialisation {
public:
    using juce::ARADocumentControllerSpecialisation::ARADocumentControllerSpecialisation;

protected:
    juce::ARAPlaybackRenderer* doCreatePlaybackRenderer() noexcept override {
        return new PassThroughRenderer(getDocumentController());
    }
    bool doRestoreObjectsFromStream(juce::ARAInputStream&, const juce::ARARestoreObjectsFilter*) noexcept override { return true; }
    bool doStoreObjectsToStream(juce::ARAOutputStream&, const juce::ARAStoreObjectsFilter*) noexcept override { return true; }
};

}  // namespace

const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory() {
    return juce::ARADocumentControllerSpecialisation::createARAFactory<SkaleDocumentController>();
}
#endif

//==============================================================================
SkaleProcessor::SkaleProcessor()
    : juce::AudioProcessor(BusesProperties()
                               .withInput("Input", juce::AudioChannelSet::stereo(), true)
                               .withOutput("Output", juce::AudioChannelSet::stereo(), true)) {
    startTimerHz(5);
}

SkaleProcessor::~SkaleProcessor() {
    stopTimer();
    jobId_.fetch_add(1);
    pool_.removeAllJobs(true, 10000);
}

void SkaleProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_.store(sampleRate);
#if JucePlugin_Enable_ARA
    prepareToPlayForARA(sampleRate, samplesPerBlock, getMainBusNumOutputChannels(), getProcessingPrecision());
#else
    juce::ignoreUnused(samplesPerBlock);
#endif
}

void SkaleProcessor::releaseResources() {
#if JucePlugin_Enable_ARA
    releaseResourcesForARA();
#endif
}

double SkaleProcessor::getTailLengthSeconds() const {
#if JucePlugin_Enable_ARA
    double tail = 0;
    if (getTailLengthSecondsForARA(tail)) return tail;
#endif
    return 0;
}

bool SkaleProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& out = layouts.getMainOutputChannelSet();
    return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo())
        && layouts.getMainInputChannelSet() == out;
}

bool SkaleProcessor::usesARA() const {
#if JucePlugin_Enable_ARA
    return isBoundToARA();
#else
    return false;
#endif
}

void SkaleProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    juce::ScopedNoDenormals noDenormals;
#if JucePlugin_Enable_ARA
    if (isBoundToARA()) {
        if (!processBlockForARA(buffer, isRealtime(), getPlayHead())) processBlockBypassed(buffer, midi);
        return;
    }
#endif
    juce::ignoreUnused(midi);
    // Sin ARA el audio pasa sin tocarse; si se está escuchando, se guarda en mono.
    if (!capturing_.load()) return;
    const juce::SpinLock::ScopedTryLockType lock(captureLock_);
    if (!lock.isLocked() || !capturing_.load()) return;
    bool hasTransport = false, playing = false;
    if (auto* head = getPlayHead()) {
        if (const auto pos = head->getPosition()) { hasTransport = true; playing = pos->getIsPlaying(); }
    }
    if (hasTransport && !playing) {
        if (heardPlayback_.load()) playbackStopped_.store(true);   // la canción ha terminado o se ha parado
        return;
    }
    if (hasTransport) heardPlayback_.store(true);
    const int n = buffer.getNumSamples(), channels = buffer.getNumChannels();
    std::size_t at = captured_.load();
    const std::size_t room = capture_.size() - std::min(at, capture_.size());
    const int count = int(std::min<std::size_t>(room, std::size_t(n)));
    for (int i = 0; i < count; ++i) {
        float s = 0.f;
        for (int c = 0; c < channels; ++c) s += buffer.getReadPointer(c)[i];
        capture_[at + std::size_t(i)] = channels ? s / float(channels) : 0.f;
    }
    captured_.store(at + std::size_t(count));
    if (count < n) playbackStopped_.store(true);   // lleno: kMaxListenSeconds
}

void SkaleProcessor::timerCallback() {
    if (capturing_.load()) {
        if (playbackStopped_.load()) {
            stopListening();
        } else {
            std::lock_guard<std::mutex> lock(stateMutex_);
            state_.listenedSeconds = double(captured_.load()) / sampleRate_.load();
        }
    }
#if JucePlugin_Enable_ARA
    if (isBoundToARA()) analyzeRegionsIfChanged();
#endif
}

#if JucePlugin_Enable_ARA
void SkaleProcessor::analyzeRegionsIfChanged() {
    // Regiones de esta instancia (la pista donde está Skale). Se vuelve a analizar si cambian.
    std::vector<juce::ARAPlaybackRegion*> regions;
    if (auto* r = getPlaybackRenderer()) regions = r->getPlaybackRegions();
    if (regions.empty()) if (auto* e = getEditorRenderer()) regions = e->getPlaybackRegions();
    std::sort(regions.begin(), regions.end(), [](auto* a, auto* b) {
        return a->getStartInPlaybackTime() < b->getStartInPlaybackTime();
    });

    juce::String sig;
    bool ready = true;
    for (auto* region : regions) {
        auto* source = region->getAudioModification()->getAudioSource();
        ready = ready && source->isSampleAccessEnabled();
        sig << juce::String::toHexString(juce::pointer_sized_int(source)) << ":" << region->getStartInAudioModificationSamples()
            << "-" << region->getEndInAudioModificationSamples() << ";";
    }
    if (!ready || sig == regionsSignature_) return;
    regionsSignature_ = sig;
    if (regions.empty()) { clear(); return; }

    const int id = jobId_.fetch_add(1) + 1;
    auto* job = new SongAnalysisJob(*this, id, regions.size() == 1 ? "Region on this track" : juce::String(int(regions.size())) + " regions on this track");
    for (auto* region : regions) {
        auto* source = region->getAudioModification()->getAudioSource();
        SongAnalysisJob::Part part;
        part.reader = std::make_unique<juce::ARAAudioSourceReader>(source);
        part.start = std::max<juce::int64>(0, region->getStartInAudioModificationSamples());
        part.end = std::min<juce::int64>(source->getSampleCount(), region->getEndInAudioModificationSamples());
        job->sampleRate = source->getSampleRate();
        if (part.end > part.start) job->parts.push_back(std::move(part));
    }
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        state_ = {};
        state_.status = SongState::Status::Working;
        state_.source = "Region on this track";
    }
    pool_.addJob(job, true);
}
#endif

SongState SkaleProcessor::state() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    SongState s = state_;
    s.progress = progress_.load();
    return s;
}

void SkaleProcessor::analyzeFile(const juce::File& f) {
    const int id = jobId_.fetch_add(1) + 1;
    capturing_.store(false);
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        state_ = {};
        state_.status = SongState::Status::Working;
        state_.source = f.getFileName();
    }
    auto* job = new SongAnalysisJob(*this, id, f.getFileName());
    job->file = f;
    pool_.addJob(job, true);
}

void SkaleProcessor::analyzeSamples(std::vector<float> mono, double sampleRate, const juce::String& source) {
    const int id = jobId_.fetch_add(1) + 1;
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        state_ = {};
        state_.status = SongState::Status::Working;
        state_.source = source;
    }
    auto* job = new SongAnalysisJob(*this, id, source);
    job->samples = std::move(mono);
    job->sampleRate = sampleRate;
    pool_.addJob(job, true);
}

void SkaleProcessor::startListening() {
    jobId_.fetch_add(1);
    capturing_.store(false);
    const juce::SpinLock::ScopedLockType lock(captureLock_);
    capture_.assign(std::size_t(kMaxListenSeconds * sampleRate_.load()), 0.f);   // reservado antes de escuchar
    captured_.store(0);
    heardPlayback_.store(false);
    playbackStopped_.store(false);
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        state_ = {};
        state_.status = SongState::Status::Listening;
        state_.source = "Song played back";
    }
    capturing_.store(true);
}

void SkaleProcessor::stopListening() {
    if (!capturing_.exchange(false)) return;
    std::vector<float> mono;
    {
        const juce::SpinLock::ScopedLockType lock(captureLock_);
        const std::size_t n = std::min(captured_.load(), capture_.size());
        mono.assign(capture_.begin(), capture_.begin() + long(n));
        std::vector<float>().swap(capture_);
    }
    analyzeSamples(std::move(mono), sampleRate_.load(), "Song played back");
}

void SkaleProcessor::analyzeAgain() {
#if JucePlugin_Enable_ARA
    if (isBoundToARA()) { regionsSignature_.clear(); return; }   // el temporizador lo vuelve a lanzar
#endif
    startListening();
}

void SkaleProcessor::clear() {
    jobId_.fetch_add(1);
    capturing_.store(false);
    std::lock_guard<std::mutex> lock(stateMutex_);
    state_ = {};
}

juce::AudioProcessorEditor* SkaleProcessor::createEditor() { return new SkaleEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SkaleProcessor(); }
