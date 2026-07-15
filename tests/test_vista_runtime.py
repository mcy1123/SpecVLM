import unittest
from types import SimpleNamespace

import torch

from vista.runtime import (
    clear_visual_selection,
    finish_score_collection,
    gather_visual_kv,
    set_visual_selection,
)


def fake_model(layer_count=3):
    layers = [SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(layer_count)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


class VistaRuntimeTest(unittest.TestCase):
    def test_set_and_clear_visual_selection(self):
        model = fake_model()
        set_visual_selection(model, torch.tensor([2, 3, 4, 5]), torch.tensor([1, 3]))
        for layer in model.model.layers:
            state = layer.self_attn.vista_visual_selection
            self.assertEqual(state["visual_positions"].tolist(), [2, 3, 4, 5])
            self.assertEqual(state["selected_positions"].tolist(), [3, 5])

        clear_visual_selection(model)
        self.assertTrue(all(
            layer.self_attn.vista_visual_selection is None
            for layer in model.model.layers
        ))

    def test_empty_implicit_route(self):
        model = fake_model(1)
        set_visual_selection(
            model,
            torch.tensor([10, 11]),
            torch.empty(0, dtype=torch.long),
        )
        selected = model.model.layers[0].self_attn.vista_visual_selection[
            "selected_positions"
        ]
        self.assertEqual(selected.numel(), 0)

    def test_finish_score_collection_reduces_boundary_scores(self):
        model = fake_model(2)
        for layer in model.model.layers:
            layer.self_attn.vista_collector_state = {
                "query_states": torch.ones(1, 2, 2, 4),
                "key_states": torch.arange(20, dtype=torch.float32).view(1, 1, 5, 4),
                "visual_positions": torch.tensor([1, 3]),
            }
            layer.self_attn.vista_collect_visual_positions = torch.tensor([1, 3])
        scores = finish_score_collection(model, boundary_index=1)
        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=5)
        self.assertTrue(all(
            layer.self_attn.vista_collector_state is None
            for layer in model.model.layers
        ))

    def test_gather_visual_kv_keeps_non_visual_and_selected_positions(self):
        keys = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
        values = keys + 10
        mask = torch.arange(6).view(1, 1, 1, 6)
        selection = {
            "visual_positions": torch.tensor([1, 2, 3]),
            "selected_positions": torch.tensor([2]),
        }
        gathered_k, gathered_v, gathered_mask, indices = gather_visual_kv(
            keys, values, mask, selection
        )
        self.assertEqual(indices.tolist(), [0, 2, 4, 5])
        self.assertEqual(gathered_k.flatten().tolist(), [0.0, 2.0, 4.0, 5.0])
        self.assertEqual(gathered_v.flatten().tolist(), [10.0, 12.0, 14.0, 15.0])
        self.assertEqual(gathered_mask.flatten().tolist(), [0, 2, 4, 5])

    def test_gather_visual_kv_dense_selection_is_identity(self):
        keys = torch.randn(1, 2, 5, 4)
        values = torch.randn(1, 2, 5, 4)
        selection = {
            "visual_positions": torch.tensor([1, 2, 3]),
            "selected_positions": torch.tensor([1, 2, 3]),
        }
        gathered_k, gathered_v, _, indices = gather_visual_kv(
            keys, values, None, selection
        )
        self.assertTrue(torch.equal(gathered_k, keys))
        self.assertTrue(torch.equal(gathered_v, values))
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
