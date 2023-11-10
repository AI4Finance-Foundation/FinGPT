def similarity_score(a, b):
    words_a = a.split()
    words_b = b.split()
    matching_words = 0

    for word_a in words_a:
        for word_b in words_b:
            if word_a in word_b or word_b in word_a:
                matching_words += 1
                break

    similarity = matching_words / min(len(words_a), len(words_b))
    return similarity