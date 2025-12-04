# V.A.R.G Images Directory

This directory contains images and screenshots displayed in the project README.

## 📸 Required Images

To populate the README gallery, add your WhatsApp images here with the following names:

### Hardware & System Photos
- `system_complete.jpg` - Photo of the complete V.A.R.G system setup
- `raspberry_pi_setup.jpg` - Close-up of Raspberry Pi Zero W with camera and OLED

### OLED Display Photos
- `oled_display_1.jpg` - OLED showing food detection in progress
- `oled_display_2.jpg` - OLED displaying calorie information
- `oled_display_3.jpg` - OLED showing system statistics

### Food Detection Examples
- `detection_example_1.jpg` - Example of food item being detected
- `detection_example_2.jpg` - Another food detection example
- `detection_example_3.jpg` - Third food detection example

## 📝 Image Guidelines

### Recommended Specifications
- **Format:** JPG or PNG
- **Resolution:** 800x600 pixels or higher
- **Orientation:** Landscape preferred for hardware shots, any for OLED/detection examples
- **File size:** Under 2MB per image (optimize if needed)

### Photo Tips
1. **Good lighting** - Natural light or well-lit environment
2. **Clean background** - Minimize clutter
3. **Clear focus** - OLED text should be readable
4. **Show the action** - Capture the system actually working

### For OLED Display Photos
- Make sure the display text is readable
- Show the detection/calorie information clearly
- Capture the transparency effect if possible

### For Food Detection Photos
- Show the camera view with food item
- Include any bounding boxes or detection overlays if your system shows them
- Show the OLED display simultaneously if possible

## 🖼️ Adding Your Images

### Option 1: From WhatsApp (Mobile)
1. Open WhatsApp on your phone
2. Select the images you want to use
3. Share → Save to Files/Photos
4. Transfer to your computer
5. Rename according to the list above
6. Place in this directory (`docs/images/`)

### Option 2: Direct Copy (Computer)
```bash
# Navigate to this directory
cd docs/images/

# Copy your images here and rename them
cp ~/Downloads/whatsapp_image_1.jpg system_complete.jpg
cp ~/Downloads/whatsapp_image_2.jpg raspberry_pi_setup.jpg
# ... and so on
```

### Option 3: Using Git
```bash
# Add images to the repository
git add docs/images/*.jpg

# Commit with a descriptive message
git commit -m "Add project images to README gallery"

# Push to repository
git push
```

## 🎨 Image Optimization (Optional)

If your images are too large, optimize them:

### Using ImageMagick
```bash
# Install ImageMagick
# macOS: brew install imagemagick
# Linux: sudo apt-get install imagemagick

# Resize and optimize
for img in *.jpg; do
  convert "$img" -resize 1200x900\> -quality 85 "optimized_$img"
done
```

### Using Python (if installed)
```python
from PIL import Image
import os

for filename in os.listdir('.'):
    if filename.endswith('.jpg'):
        img = Image.open(filename)
        # Resize if too large
        if img.size[0] > 1200:
            img.thumbnail((1200, 900))
        # Save with optimization
        img.save(filename, optimize=True, quality=85)
```

## 📂 Current Status

Check which images you've added:
```bash
ls -lh docs/images/*.jpg
```

## 🔗 Where These Images Appear

These images are displayed in:
- Main `README.md` - Gallery section at the top
- GitHub repository page
- Project documentation

## ⚠️ Important Notes

1. **Copyright:** Only use images you have the rights to use
2. **Privacy:** Don't include identifiable information (faces, personal data)
3. **File names:** Use exact names listed above for automatic display
4. **Git LFS:** For very large images (>10MB), consider using Git LFS
5. **Placeholder:** If an image is missing, GitHub will show a broken image icon

## 🆘 Troubleshooting

### Images not showing in README?
1. Check file names match exactly (case-sensitive)
2. Ensure images are committed and pushed to git
3. Verify images are in correct directory: `docs/images/`
4. Check image format is supported (JPG, PNG, GIF)

### Images too large?
- Use image optimization tools above
- Consider converting to JPG with lower quality
- Resize to reasonable dimensions (800-1200px width)

## ✅ Checklist

- [x] `system_complete.jpg` added ✓
- [ ] `raspberry_pi_setup.jpg` (optional)
- [x] `oled_display_1.jpg` added ✓
- [ ] `oled_display_2.jpg` (optional)
- [ ] `oled_display_3.jpg` (optional)
- [ ] `detection_example_1.jpg` (optional)
- [ ] `detection_example_2.jpg` (optional)
- [ ] `detection_example_3.jpg` (optional)
- [x] All images optimized and under 2MB ✓
- [ ] Images committed to git
- [ ] README displays correctly on GitHub

**Current Status:** 2 images added and configured in README!

---

**Need help?** Check the main [README.md](../../README.md) for project documentation.

