from bs4 import BeautifulSoup

# Load your HTML file
with open("tum.html", "r", encoding="utf-8") as file:
    html = file.read()

soup = BeautifulSoup(html, 'html.parser')

# Locate the div with the lyrics
lyrics_div = soup.find("div", class_="text-base lg:text-lg pb-2 text-center md:text-left")

# Extract text cleanly, preserving line breaks
lyrics = ""
if lyrics_div:
    for p in lyrics_div.find_all('p'):
        text = p.get_text(separator="\n")  # Converts <br/> to \n automatically
        lyrics += text.strip() + "\n\n"
else:
    print("Lyrics div not found!")

# Save to file or print
with open("lyrics.txt", "w", encoding="utf-8") as f:
    f.write(lyrics.strip())

print(lyrics.strip())
