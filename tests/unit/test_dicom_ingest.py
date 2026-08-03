"""Synthetic DICOM ingest coverage for every rejection reason."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from echosynth.data.dicom_ingest import ingest

STANDARD = "1.2.840.10008.5.1.4.1.1.6.1"


def _write_dicom(
    path: Path,
    *,
    sop_class: str = STANDARD,
    sop_instance: str | None = None,
    color: bool = False,
    pixel_data: bool = True,
    broken_pixels: bool = False,
) -> str:
    instance = sop_instance or generate_uid()
    metadata = FileMetaDataset()
    metadata.MediaStorageSOPClassUID = sop_class
    metadata.MediaStorageSOPInstanceUID = instance
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = instance
    dataset.PatientID = "fixture-patient"
    dataset.StudyInstanceUID = generate_uid()
    dataset.ClinicalTrialSubjectReadingID = "CA"
    dataset.StudyDate = "2020" + "0101"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.NumberOfFrames = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    if color:
        dataset.UltrasoundColorDataPresent = 1
    if pixel_data:
        array = np.stack([np.arange(64, dtype=np.uint8).reshape(8, 8)] * 2)
        dataset.PixelData = b"broken" if broken_pixels else array.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return instance


def test_ingest_reason_buckets_and_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "dicoms"
    root.mkdir()
    instance = _write_dicom(root / "a-valid.dcm")
    _write_dicom(root / "b-color.dcm", color=True)
    _write_dicom(root / "c-nonstandard.dcm", sop_class=generate_uid())
    _write_dicom(root / "d-no-pixels.dcm", pixel_data=False)
    (root / "e-corrupt.dcm").write_bytes(b"not a dicom")
    _write_dicom(root / "f-duplicate.dcm", sop_instance=instance)
    _write_dicom(root / "g-decode.dcm", broken_pixels=True)
    config = tmp_path / "config.yaml"
    config.write_text(f"standard_sop_class_uids: ['{STANDARD}']\n", encoding="utf-8")
    out = tmp_path / "images"
    manifest = tmp_path / "manifest.csv"
    result = ingest(root, out, manifest, 2, config_path=config, dry_run=True)
    assert len(result) == 1
    assert result.attrs["rejections"] == {
        "COLOR_DOPPLER": 1,
        "DECODE_FAILED": 1,
        "DUPLICATE_ID": 1,
        "NONSTANDARD_SOP": 1,
        "NO_PIXEL_DATA": 1,
        "UNREADABLE": 1,
    }
    assert not out.exists()
    assert not manifest.exists()


def test_ingest_writes_one_image_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dicoms"
    root.mkdir()
    _write_dicom(root / "valid.dcm")
    config = tmp_path / "config.yaml"
    config.write_text(f"standard_sop_class_uids: ['{STANDARD}']\n", encoding="utf-8")
    result = ingest(
        root,
        tmp_path / "images",
        tmp_path / "manifest.csv",
        1,
        config_path=config,
    )
    assert len(result) == 1
    assert (tmp_path / "manifest.csv").is_file()
    assert len(list((tmp_path / "images").glob("*.png"))) == 1
