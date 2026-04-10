from wordfreq import zipf_frequency, top_n_list

# Get top 50,000 words
words = top_n_list("en", 50000)

# Save to corpus.txt
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(words))

print("✅ corpus.txt created successfully!")