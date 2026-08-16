import unittest
import torch

from eval.non_linear_replacement import (
    PReLU_Approx,
    calibrate_resnet_activations_with_poly,
    replace_resnet_activations_with_poly,
    replace_resnet_activations_with_poly_scales,
)


class SmallPReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.PReLU(1, init=0.25)
        self.expand = torch.nn.Linear(2, 2, bias=False)
        self.second = torch.nn.PReLU(1, init=0.1)
        with torch.no_grad():
            self.expand.weight.copy_(torch.tensor([[3.0, -1.0], [2.0, 4.0]]))

    def forward(self, inputs):
        return self.second(self.expand(self.first(inputs)))


class NonLinearReplacementTest(unittest.TestCase):
    def test_layerwise_calibration_uses_partially_converted_input_ranges(self):
        model = SmallPReLUModel().eval()
        inputs = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])

        diagnostics = calibrate_resnet_activations_with_poly(
            model, inputs, scale_margin=2.0)

        self.assertEqual(
            [item['module'] for item in diagnostics], ['first', 'second'])
        self.assertAlmostEqual(diagnostics[0]['input_scale'], 4.0)
        self.assertGreater(
            diagnostics[1]['input_absmax'], diagnostics[0]['input_absmax'])
        self.assertTrue(all(
            isinstance(module, PReLU_Approx)
            for module in (model.first, model.second)
        ))
        self.assertTrue(torch.isfinite(model(inputs)).all())

    def test_fixed_layer_scales_require_an_exact_name_mapping(self):
        model = SmallPReLUModel().eval()
        with self.assertRaisesRegex(ValueError, 'missing'):
            replace_resnet_activations_with_poly_scales(
                model, {'first': 2.0})

        replaced = replace_resnet_activations_with_poly_scales(
            model, {'first': 2.0, 'second': 8.0})

        self.assertEqual(replaced, 2)
        self.assertAlmostEqual(float(model.first.relu.input_scale), 2.0)
        self.assertAlmostEqual(float(model.second.relu.input_scale), 8.0)

    def test_layerwise_calibration_rejects_nonfinite_inputs(self):
        model = SmallPReLUModel().eval()
        inputs = torch.tensor([[float('nan'), 0.0]])

        with self.assertRaisesRegex(FloatingPointError, 'Calibration inputs'):
            calibrate_resnet_activations_with_poly(model, inputs)

    def test_replacement_stays_on_the_source_module_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SmallPReLUModel().to(device).eval()

        replaced = replace_resnet_activations_with_poly(model, input_scale=8.0)

        self.assertEqual(replaced, 2)
        model_device = next(model.parameters()).device
        self.assertEqual(model.first.slope.device, model_device)
        self.assertEqual(model.first.relu.input_scale.device, model_device)


if __name__ == '__main__':
    unittest.main()
