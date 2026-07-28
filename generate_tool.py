import os
import random
import urllib.request
import xml.etree.ElementTree as ET
from google import genai

def get_trending_topic():
    # Pulls the live daily trending searches from Google for free
    url = 'https://trends.google.com/trends/trendingsearches/daily/rss?geo=US'
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        # Extract the trending keywords
        trends = [item.text for item in root.findall('.//item/title')]
        
        if trends:
            # Pick a random trend from the top 15
            return random.choice(trends[:15]) 
    except Exception as e:
        print(f"Could not fetch trends: {e}")
    
    return "Freelance Tax Calculator" # Fallback just in case

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
        
    client = genai.Client(api_key=api_key)
    
    # 1. Get today's real Google Trend
    topic = get_trending_topic()
    print(f"Selected real-time trend: {topic}")
    
    # 2. Ask Gemini to invent a tool based on the trend
    prompt = f"""
    The current trending topic on Google Search is '{topic}'. 
    Brainstorm a simple, single-page utility web app (like a calculator, tracker, checklist, or converter) that someone interested in this topic might find useful.
    Then, write the complete HTML file for this app. Include Tailwind CSS via CDN for modern styling, and write all JavaScript logic inside the HTML file.
    Output ONLY the raw HTML code. Do not include markdown formatting like ```html.
    """
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    
    # Clean up the output to ensure it's pure HTML
    html_content = response.text.replace("```html", "").replace("```", "").strip()
    
    # 3. Save it to the app folder for Vercel
    os.makedirs("app", exist_ok=True)
    with open("app/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Successfully generated app/index.html based on a live trend!")

if __name__ == "__main__":
    main()
