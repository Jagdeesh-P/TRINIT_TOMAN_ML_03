from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Image
from .ml_model import generate_caption

def upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['file']
            caption = generate_caption(image)
            Image.objects.create(file=image, caption=caption)
            return render(request, 'result.html', {'image': image, 'caption': caption})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
