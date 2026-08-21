"""Unit tests for 26_rekognition_enrich.py's pure _flag() label-presence
scorer -- no network calls, no AWS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
rek = importlib.import_module("26_rekognition_enrich")


def test_flag_zero_when_no_label_in_vocab():
    labels = [("Tree", 0.9), ("Sky", 0.8)]
    assert rek._flag(labels, rek.WEAPON) == 0.0


def test_flag_picks_highest_confidence_match():
    labels = [("Knife", 0.6), ("Sword", 0.95), ("Tree", 0.9)]
    assert rek._flag(labels, rek.WEAPON) == 0.95


def test_flag_case_insensitive():
    labels = [("GUN", 0.7)]
    assert rek._flag(labels, rek.WEAPON) == 0.7


def test_flag_empty_labels_list():
    assert rek._flag([], rek.ANIMAL) == 0.0


def test_weapon_vocab_matches_real_examples():
    assert "knife" in rek.WEAPON
    assert "chainsaw" in rek.WEAPON
    assert "flower" not in rek.WEAPON


def test_animal_vocab_matches_real_examples():
    assert "shark" in rek.ANIMAL
    assert "wolf" in rek.ANIMAL


def test_person_vocab_matches_real_examples():
    assert "person" in rek.PERSON
    assert "child" in rek.PERSON


def test_multiple_vocabs_independent():
    labels = [("Knife", 0.5), ("Dog", 0.9), ("Fire", 0.3)]
    assert rek._flag(labels, rek.WEAPON) == 0.5
    assert rek._flag(labels, rek.ANIMAL) == 0.9
    assert rek._flag(labels, rek.FIRE) == 0.3
    assert rek._flag(labels, rek.WATER) == 0.0
