from transformers import pipeline

# Load summarization pipeline 
summerizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text sample
text = """
India’s Chandrayaan-3 mission successfully landed near the Moon’s south pole, a region previously unexplored by other missions.
This marks a major milestone for India's space program. Scientists expect to study the Moon’s surface for water ice and other resources.
The mission also includes a rover to explore the terrain.
"""

#Generate the summary
summary = summerizer(text, max_length=60, min_length=25, do_sample=False)

print("Summary:\n:", summary[0]['summary_text'])