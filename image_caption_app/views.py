# views.py
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Image
from .ml_model import generate_caption  # Import your caption generation function

def upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('result')
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})

def result(request):
    latest_image = Image.objects.last()  # Assuming you want to display the latest uploaded image
    caption = generate_caption(latest_image.file)  # Generate caption for the latest image
    return render(request, 'result.html', {'image': latest_image, 'caption': caption})

