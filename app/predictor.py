import torch
from torchvision import transforms
from PIL import Image
import io
import base64

class PokemonPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.classes = [
            "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", 
            "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice", 
            "Normal", "Poison", "Psychic", "Rock", "Steel", "Water"
        ]

    def preprocess(self, base64_data):
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, base64_data):
        try:
            tensor = self.preprocess(base64_data)
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item()
                
            return {
                "type": self.classes[predicted.item()],
                "confidence": round(confidence, 4)
            }
        except (TypeError, base64.binascii.Error):
            raise ValueError("Format base64 tidak valid atau data korup")
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan pemrosesan gambar: {str(e)}")