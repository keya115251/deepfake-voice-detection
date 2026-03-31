from utils.feature_extractor import DualFeatureExtractor

# Load ONCE (very important for performance)
extractor = DualFeatureExtractor()


def extract_features(file_path):
    mfcc, wav2vec = extractor.extract(file_path)

    return mfcc, wav2vec