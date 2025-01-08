from PIL import Image
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

class LabelStorage:
    def __init__(self):
        self.images = []

    def add_image(self, image_bytes: bytes):
        try:
            image = BytesIO(image_bytes)
            self.images.append(image)
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

    def _create_composite_image(self) -> Image:
        if not self.images:
            raise ValueError("No images to merge.")

        # Get dimensions of images
        widths, heights = zip(*(img.size for img in self.images))

        total_height = sum(heights)
        max_width = max(widths)

        # Create a new blank image with the appropriate size
        composite_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in self.images:
            composite_image.paste(img, (0, y_offset))
            y_offset += img.height

        return composite_image
    
    def _create_pdf_document(self) -> BytesIO:
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)

        for image in self.images:
            # Convert PIL image to bytes
            img_buffer = ImageReader(image)
            # image.save(img_buffer, format='PNG')
            # img_buffer.seek(0)

            # Add image to the PDF page
            c.drawImage(image=img_buffer, x=0, y=0, width=letter[0], height=letter[1])
            # c.showPage()  # End the current page and start a new one

        c.save()
        pdf_buffer.seek(0)

        return pdf_buffer
    
    def clear(self):
        self.images = []

    def get_document(self, format='pdf') -> bytes:
        # Ensure there are images to merge
        if not self.images:
            raise ValueError("No images to merge.")
        
        output = BytesIO()
        
        if format == 'pdf':
            output = self._create_pdf_document()
        elif format == 'png':
            composite_image = self._create_composite_image()
            composite_image.save(output, format='PNG')
        else:
            raise ValueError("Unknown document format output.")

        return output.getvalue()
