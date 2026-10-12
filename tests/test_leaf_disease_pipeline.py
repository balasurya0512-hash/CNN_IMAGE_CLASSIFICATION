import os
import unittest


class LeafDiseasePipelineTests(unittest.TestCase):
    def test_dataset_folders_exist(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Computer vision'))
        self.assertTrue(os.path.isdir(root))
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.assertGreaterEqual(len(classes), 3)

    def test_model_functions_are_available(self):
        import importlib.util

        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'leaf_disease_cnn.py'))
        self.assertTrue(os.path.exists(module_path), 'leaf_disease_cnn.py should exist in the project root')

        spec = importlib.util.spec_from_file_location('leaf_disease_cnn', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, 'build_cnn_model'))
        self.assertTrue(hasattr(module, 'load_image_dataset'))


if __name__ == '__main__':
    unittest.main()
