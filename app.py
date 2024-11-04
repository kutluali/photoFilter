import cv2
import numpy as np
import gradio as gr


def apply_filter(filter_type, input_image, blur_amount, brightness, contrast, custom_text):
    if input_image is None:
        return None

   
    input_image = cv2.convertScaleAbs(input_image, alpha=contrast, beta=brightness)

   
    if blur_amount > 0:
        blur_amount = max(1, int(blur_amount))
        if blur_amount % 2 == 0:
            blur_amount += 1
        input_image = cv2.GaussianBlur(input_image, (blur_amount, blur_amount), 0)

    # Filtre uygulama
    if filter_type == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        input_image = cv2.filter2D(input_image, -1, kernel)
    elif filter_type == "Edge Detection":
        input_image = cv2.Canny(input_image, 100, 200)
    elif filter_type == "Invert":
        input_image = cv2.bitwise_not(input_image)
    elif filter_type == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        input_image = cv2.transform(input_image, sepia_filter)
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)
    elif filter_type == "Negative":
        input_image = cv2.bitwise_not(input_image)
    elif filter_type == "Grayscale":
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "Emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        input_image = cv2.filter2D(input_image, -1, kernel)
    elif filter_type == "Sketch":
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray_image)
        blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
        input_image = cv2.divide(gray_image, 255 - blurred, scale=256)
    
    
    if filter_type in ["WANTED Effect", "Kendin Yaz"]:
        height, width = input_image.shape[:2]
        y_position = int(height * 0.85)  # %15 yukarı
        text = "WANTED" if filter_type == "WANTED Effect" else custom_text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_color = (0, 0, 255)  # Kırmızı renk
        line_color = (0, 0, 0)    # Siyah çizgi

        # Metni ortalamak için metnin boyutunu hesapla
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        x_position = (width - text_size[0]) // 2

        # Alt siyah çizgi
        cv2.line(input_image, (0, y_position), (width, y_position), line_color, thickness=3)
        # Kırmızı yazıyı çizginin hemen üstüne ekle
        cv2.putText(input_image, text, (x_position, y_position - 10), font, font_scale, text_color, thickness=font_thickness)

    return input_image

# Görüntüyü kaydetme fonksiyonu
def save_image(output_image):
    if output_image is not None:
        filename = "output_image.jpg"
        cv2.imwrite(filename, output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return f"Görüntü kaydedildi: {filename}"
    return "Kaydedilecek görüntü yok."

# Sayfayı sıfırlama fonksiyonu
def reset_page():
    return None, None, "Gaussian Blur", 0, 0, 1.0, ""

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# Web Kameradan Canlı Filtreleme")

    # Filtre seçenekleri
    filter_type = gr.Dropdown(
        label="Filtre Seçin",
        choices=["Gaussian Blur", "Sharpen", "Edge Detection", "Invert", "Sepia", "Negative", "Grayscale", "Emboss", "Sketch", "WANTED Effect", "Kendin Yaz"],
        value="Gaussian Blur",
        interactive=True
    )

   
    custom_text_input = gr.Textbox(label="Kendi Yazınızı Girin", placeholder="Buraya yazın", visible=False)

    # Kaydırıcılar
    blur_amount = gr.Slider(minimum=0, maximum=50, label="Bulanıklık Miktarı", value=0)
    brightness = gr.Slider(minimum=-100, maximum=100, label="Parlaklık", value=0)
    contrast = gr.Slider(minimum=0.5, maximum=3.0, step=0.1, label="Konsantras (Kontrast)", value=1.0)

    # Giriş ve çıkış görüntüleri için yan yana düzenleme
    with gr.Row():
        input_image = gr.Image(label="Resim Yükle", type="numpy")
        output_image = gr.Image(label="Filtre Uygulandı", type="numpy", interactive=False)

    # Filtre uygula butonu
    apply_button = gr.Button("Filtreyi Uygula")

    # Kaydetme butonu
    save_button = gr.Button("Görüntüyü İndir")
    save_output = gr.Textbox(label="İndirme Durumu")

    # Geri dön butonu
    reset_button = gr.Button("Sayfayı Sıfırla")

    # "Kendin Yaz" seçeneği seçildiğinde metin kutusunu göster
    def toggle_custom_text(filter_type):
        return gr.update(visible=filter_type == "Kendin Yaz")

    filter_type.change(fn=toggle_custom_text, inputs=filter_type, outputs=custom_text_input)

    # Butona tıklanınca filtre uygulama fonksiyonu
    apply_button.click(
        fn=apply_filter,
        inputs=[filter_type, input_image, blur_amount, brightness, contrast, custom_text_input],
        outputs=output_image
    )

    # Kaydetme butonu
    save_button.click(fn=save_image, inputs=output_image, outputs=save_output)

    # Geri dön butonu
    reset_button.click(
        fn=reset_page,
        inputs=None,
        outputs=[input_image, output_image, filter_type, blur_amount, brightness, contrast, custom_text_input]
    )


demo.launch()
