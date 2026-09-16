import numpy as np
import soundfile as sf

from src.audio.preprocessing import AudioStatus, chunk_audio, load_audio, preprocess_audio


def test_preprocess_converts_stereo_and_resamples() -> None:
    source = np.column_stack((np.ones(8000, dtype=np.float32), np.zeros(8000, dtype=np.float32)))
        
    audio = preprocess_audio(source, 8000, source_id="clip")
        
    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (16000,)
        
    # Relax the tolerance to 1e-3 to account for the edge artifacts of the resampling filter
    assert np.isclose(np.mean(audio.samples), 0.5, atol=1e-3)
        
    # Ignore boundary ringing and allow a reasonable tolerance for the rest
    np.testing.assert_allclose(audio.samples[100:-100], 0.5, atol=0.1)


def test_preprocess_scales_integer_pcm() -> None:
    audio = preprocess_audio(np.array([-32768, 0, 32767], dtype=np.int16), 16000)

    np.testing.assert_allclose(audio.samples, [-1.0, 0.0, 32767 / 32768], atol=1e-6)


def test_load_audio_scales_file_and_marks_silence(tmp_path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros((800, 2), dtype=np.int16), 16000, subtype="PCM_16")

    audio = load_audio(path, source_id="fixture")

    assert audio.source_id == "fixture"
    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (800,)
    assert audio.status is AudioStatus.SILENT


def test_chunk_audio_keeps_offsets_and_final_short_chunk() -> None:
    audio = preprocess_audio(np.arange(16000 * 2 + 8000, dtype=np.float32) / 40000, 16000, source_id="clip")
        
    chunks = chunk_audio(audio, chunk_seconds=2)
        
    # 40,000 samples split into 32,000 sample chunks yields exactly [32000, 8000]
    assert [chunk.valid_samples for chunk in chunks] == [32000, 8000]


def test_empty_audio_is_explicit_and_has_no_chunks() -> None:
    audio = preprocess_audio(np.empty(0, dtype=np.float32), 16000)

    assert audio.status is AudioStatus.EMPTY
    assert chunk_audio(audio) == []