from ballistics_model import BallisticsModel
from data_processor import BallisticsDataProcessor
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Ballistics Recognition System')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True)
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    # Initialize model and data processor
    model = BallisticsModel()
    data_processor = BallisticsDataProcessor(args.data_dir)

    if args.mode == 'train':
        # Load and preprocess data
        X, y = data_processor.load_dataset()
        X_train, X_test, y_train, y_test = data_processor.split_dataset(X, y)
        
        # Train model
        model.train(X_train, y_train)
        
        # Save model using pickle
        with open('ballistics_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved successfully!")
        
    elif args.mode == 'predict':
        if not args.image_path:
            print("Please provide an image path for prediction")
            return
            
        # Load model and make prediction
        predicted_class, confidence = model.predict(args.image_path)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main() 