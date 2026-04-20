from pyngrok import ngrok
import time

tunnel = ngrok.connect(8000)
print("Public URL:", tunnel.public_url)
print("Share this URL with anyone!")
print("Press Ctrl+C to stop")

while True:
    time.sleep(60)