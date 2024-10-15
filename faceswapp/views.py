from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
from PIL import Image
import cv2
import insightface
from insightface.app import FaceAnalysis
from io import BytesIO
import os
from django.conf import settings

# Initialize FaceAnalysis and use the locally stored model
app = FaceAnalysis(name="buffalo_l", root="", download=False)
app.prepare(ctx_id=0, det_size=(640, 640))

model_path = os.path.join(settings.BASE_DIR, 'faceswapp', 'models', 'inswapper_128.onnx')

swapper = insightface.model_zoo.get_model(model_path, download=False)  # Use local model without downloading

def face_swap_view(request):
    if request.method == 'POST' and request.FILES.get('poster') and request.FILES.get('facial'):
        # Open images with PIL and convert to NumPy array in RGB format
        poster = Image.open(request.FILES['poster']).convert('RGB')
        facial = Image.open(request.FILES['facial']).convert('RGB')

        # Convert PIL images to NumPy arrays
        poster_array = np.array(poster)
        facial_array = np.array(facial)

        poster_faces = app.get(poster_array)
        facial_faces = app.get(facial_array)

        if not poster_faces or not facial_faces:
            return render(request, 'result.html', {'error': 'No faces detected in one or both images.'})

        facial_face = facial_faces[0]
        result = poster_array.copy()

        for face in poster_faces:
            result = swapper.get(result, face, facial_face, paste_back=True)

        # Convert NumPy array back to an image and save it in memory
        swapped_image = Image.fromarray(result)
        buffer = BytesIO()
        swapped_image.save(buffer, format='JPEG')
        buffer.seek(0)

        # Create an InMemoryUploadedFile object to pass the image to the template
        swapped_image_file = InMemoryUploadedFile(buffer, None, 'swapped_image.jpg', 'image/jpeg', buffer.getbuffer().nbytes, None)

        # Return the image directly without saving to disk
        return render(request, 'result.html', {'swapped_image_file': swapped_image_file})

    else:
        return render(request, 'index.html')
