/* The tokenizer against token IDs taken from transformers (AutoTokenizer,
 * add_special_tokens=False).
 *
 *   h3_tokenizer_tests [<released tokenizer.json>]
 *
 * Always: a small vocabulary written to a temporary folder, with and
 * without a tokenizer_config.json that adds special tokens beside it. With
 * the released Qwen vocabulary too: its own cases, and MiniMax-H3's tags
 * (<d>, </d> and five more, which only its tokenizer_config.json lists) -
 * as special tokens where that file, listing them, is beside
 * tokenizer.json, as plain text where it is not (an image model's folder). */
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void check_case(h3_tokenizer *tokenizer, const char *text,
                       const uint32_t *expected, size_t expected_count) {
    char error[256];
    uint32_t *ids = NULL;
    size_t count = 0;
    CHECK(h3_tokenizer_encode(tokenizer, text, 1, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == expected_count);
    for (size_t index = 0; index < count; index++) CHECK(ids[index] == expected[index]);
    char *decoded = h3_tokenizer_decode(tokenizer, ids, count, error, sizeof(error));
    CHECK(decoded != NULL);
    CHECK(!strcmp(decoded, text));
    free(decoded);
    h3_tokenizer_ids_free(ids);
}

#define CASE(tokenizer, text, ...) do { \
    const uint32_t expected_ids[] = {__VA_ARGS__}; \
    check_case(tokenizer, text, expected_ids, \
               sizeof(expected_ids) / sizeof(expected_ids[0])); \
} while (0)

static void write_text(const char *folder, const char *name, const char *text) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", folder, name);
    FILE *file = fopen(path, "w");
    CHECK(file != NULL);
    CHECK(fputs(text, file) >= 0);
    CHECK(fclose(file) == 0);
}

static h3_tokenizer *load_folder(const char *folder) {
    char path[1200], error[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", folder);
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    if (!tokenizer) fprintf(stderr, "tokenizer: %s\n", error);
    return tokenizer;
}

static void remove_folder(const char *folder) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/tokenizer.json", folder);
    unlink(path);
    snprintf(path, sizeof(path), "%s/tokenizer_config.json", folder);
    unlink(path);
    rmdir(folder);
}

/* 19 symbols (the space is the byte-level U+0120), two merges, and the
 * added tokens 19 and 20 (the last one's ID is %d). */
#define SMALL_TOKENIZER \
    "{\"version\":\"1.0\",\"added_tokens\":[" \
    "{\"id\":19,\"content\":\"<|endoftext|>\",\"single_word\":false," \
    "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}," \
    "{\"id\":%d,\"content\":\"<|im_start|>\",\"single_word\":false," \
    "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}]," \
    "\"normalizer\":{\"type\":\"NFC\"}," \
    "\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"vocab\":{" \
    "\"H\":0,\"e\":1,\"l\":2,\"o\":3,\"\\u0120\":4,\"<\":5,\"d\":6,\">\":7," \
    "\"/\":8,\"s\":9,\"a\":10,\"y\":11,\"He\":12,\"ll\":13,\"|\":14,\"c\":15," \
    "\"u\":16,\"t\":17,\"f\":18},\"merges\":[\"H e\",\"l l\"]}}"

/* Special tokens a tokenizer_config.json adds beside a tokenizer.json that
 * does not have them. The IDs are what transformers 5.23 gives for these
 * files (with Qwen's pre-tokenizer and byte-level decoder added to
 * tokenizer.json, which this tokenizer has built in). */
static void small_vocabulary(void) {
    char folder[1024], text[4096];
    const char *tmp = getenv("TMPDIR");
    snprintf(folder, sizeof(folder), "%s/h3-tokenizer-test-XXXXXX",
             tmp && tmp[0] ? tmp : "/tmp");
    CHECK(mkdtemp(folder) != NULL);
    snprintf(text, sizeof(text), SMALL_TOKENIZER, 20);
    write_text(folder, "tokenizer.json", text);

    /* tokenizer.json alone: the tags are text. */
    h3_tokenizer *tokenizer = load_folder(folder);
    CHECK(tokenizer != NULL);
    CASE(tokenizer, "He says Hello", 12, 4, 9, 10, 11, 9, 4, 12, 13, 3);
    CASE(tokenizer, "He says <d>Hello</d>", 12, 4, 9, 10, 11, 9, 4, 5, 6,
         7, 12, 13, 3, 5, 8, 6, 7);
    CASE(tokenizer, "<|im_start|>", 20);
    h3_tokenizer_free(tokenizer);

    /* A list that names one token tokenizer.json has (it keeps its ID) and
     * three it does not: they get 21, 22 and 23, in the order listed. */
    write_text(folder, "tokenizer_config.json",
               "{\"additional_special_tokens\":[\"<|im_start|>\",\"<d>\","
               "\"</d>\",\"<|cutoff|>\"],\"tokenizer_class\":\"Qwen2Tokenizer\"}");
    tokenizer = load_folder(folder);
    CHECK(tokenizer != NULL);
    CASE(tokenizer, "He says Hello", 12, 4, 9, 10, 11, 9, 4, 12, 13, 3);
    /* The space before the tag stays, a token of its own. */
    CASE(tokenizer, "He says <d>Hello</d>", 12, 4, 9, 10, 11, 9, 4, 21,
         12, 13, 3, 22);
    CASE(tokenizer, "<d>Hello</d><d>Hello</d>", 21, 12, 13, 3, 22, 21, 12,
         13, 3, 22);
    CASE(tokenizer, "says<d> Hello </d>says", 9, 10, 11, 9, 21, 4, 12, 13,
         3, 4, 22, 9, 10, 11, 9);
    CASE(tokenizer, "<|im_start|><|cutoff|><|endoftext|>", 20, 23, 19);
    /* Not the tags: another case, a space inside. */
    CASE(tokenizer, "<d >Hello</ d>", 5, 6, 4, 7, 12, 13, 3, 5, 8, 4, 6, 7);
    h3_tokenizer_free(tokenizer);

    /* transformers 5 writes the list as extra_special_tokens. */
    write_text(folder, "tokenizer_config.json",
               "{\"extra_special_tokens\":[\"<d>\",\"</d>\"]}");
    tokenizer = load_folder(folder);
    CHECK(tokenizer != NULL);
    CASE(tokenizer, "He says <d>Hello</d>", 12, 4, 9, 10, 11, 9, 4, 21,
         12, 13, 3, 22);
    h3_tokenizer_free(tokenizer);

    /* A list with nothing new, and a file that is not JSON: as without. */
    write_text(folder, "tokenizer_config.json",
               "{\"additional_special_tokens\":[\"<|im_start|>\"]}");
    tokenizer = load_folder(folder);
    CHECK(tokenizer != NULL);
    CASE(tokenizer, "<d>", 5, 6, 7);
    h3_tokenizer_free(tokenizer);
    write_text(folder, "tokenizer_config.json", "not JSON");
    tokenizer = load_folder(folder);
    CHECK(tokenizer != NULL);
    CASE(tokenizer, "<d>", 5, 6, 7);
    h3_tokenizer_free(tokenizer);

    /* IDs with a gap (19, then 30): the next free ID is not the count of
     * tokens, so a new token is refused, not given an ID in use. */
    snprintf(text, sizeof(text), SMALL_TOKENIZER, 30);
    write_text(folder, "tokenizer.json", text);
    write_text(folder, "tokenizer_config.json",
               "{\"additional_special_tokens\":[\"<d>\"]}");
    char path[1200], error[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", folder);
    CHECK(h3_tokenizer_load(path, error, sizeof(error)) == NULL);
    CHECK(error[0] != '\0');
    remove_folder(folder);
}

/* MiniMax-H3's tags in the released vocabulary. */
static void released_tags(h3_tokenizer *tokenizer, const char *path) {
    /* Like the tags, and text in every folder. */
    CASE(tokenizer, "He types a < b and x<d on the board, </div> tags.",
         1519, 4494, 264, 366, 293, 323, 856, 90707, 389, 279, 4479, 11,
         690, 611, 29, 9492, 13);
    CASE(tokenizer, "雪の中を歩く赤い狐。カメラはゆっくり引いていく。",
         100167, 125732, 29412, 143852, 46784, 104565, 16586, 100324, 1773,
         136539, 15322, 142034, 72586, 124141, 124733, 1773);

    char config[2048];
    const char *slash = strrchr(path, '/');
    snprintf(config, sizeof(config), "%.*s%s",
             slash ? (int)(slash - path + 1) : 0, path, "tokenizer_config.json");
    static char listed[1 << 16];
    FILE *file = fopen(config, "r");
    size_t length = file ? fread(listed, 1, sizeof(listed) - 1, file) : 0;
    if (file) fclose(file);
    listed[length] = '\0';
    if (!strstr(listed, "\"<d>\"")) {
        /* An image model's folder: no tags. */
        CASE(tokenizer, "He says <d>Hello</d>", 1519, 2727, 366, 67, 79497,
             522, 67, 29);
        printf("note: no tokenizer_config.json with MiniMax-H3's tags beside "
               "the tokenizer; they are checked as plain text\n");
        return;
    }
    CASE(tokenizer, "He says <d>Hello</d>", 1519, 2727, 220, 151669, 9707,
         151670);
    CASE(tokenizer, "<d>Hello</d>", 151669, 9707, 151670);
    CASE(tokenizer, "A man whispers<d> Good night </d>and leaves.", 32, 883,
         88148, 151669, 7684, 3729, 220, 151670, 437, 10901, 13);
    CASE(tokenizer, "彼女は<d>ありがとう</d>と言う。", 101429, 57750, 15322,
         151669, 85690, 151670, 131622, 1773);
    CASE(tokenizer, "A woman looks at the camera and says "
         "<d>こんにちは、今日はいい天気ですね。</d> She smiles.", 32, 5220,
         5868, 518, 279, 6249, 323, 2727, 220, 151669, 89015, 5373, 133165,
         126224, 35727, 94121, 128797, 1773, 151670, 2932, 42532, 13);
    CASE(tokenizer, "He says\n<d>Hi</d>\n\nBye", 1519, 2727, 198, 151669,
         13048, 151670, 271, 1359, 68);
    CASE(tokenizer, "Music plays. <|lyrics_start|>La la la<|lyrics_end|> "
         "<|caption_start|>soft piano<|caption_end|><|cutoff|>", 24128,
         11088, 13, 220, 151672, 8747, 1187, 1187, 151673, 220, 151674,
         3510, 26278, 151675, 151671);
    /* Only the exact spelling is a tag. */
    CASE(tokenizer, "He says <d>Hello</d >, then <D>shouts</D> and < d >.",
         1519, 2727, 220, 151669, 9707, 522, 67, 861, 11, 1221, 366, 35, 29,
         927, 11672, 522, 35, 29, 323, 366, 294, 861, 13);
}

int main(int argc, char **argv) {
    small_vocabulary();
    if (argc < 2) {
        printf("ok: %d tokenizer checks (small vocabulary)\n", checks);
        return 0;
    }
    const char *path = argv[1];
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "FAIL tests/test_tokenizer.c: %s\n", error);
        return 1;
    }
    released_tags(tokenizer, path);
    const uint32_t fox[] = {32, 2518, 38835, 11435, 1526, 11794};
    check_case(tokenizer, "A red fox walking through snow", fox,
               sizeof(fox) / sizeof(fox[0]));
    const uint32_t unicode[] = {9707, 11, 50891, 0, 220, 220, 17, 15, 17,
                                21, 198, 104811, 51950, 594};
    check_case(tokenizer, "Hello, WORLD!  2026\n中文 café's", unicode,
               sizeof(unicode) / sizeof(unicode[0]));
    const uint32_t emoji[] = {145080, 64, 4894};
    check_case(tokenizer, "🙂a!\n", emoji,
               sizeof(emoji) / sizeof(emoji[0]));
    const uint32_t special[] = {151644};
    check_case(tokenizer, "<|im_start|>", special, 1);
    const uint32_t cinematic[] = {32, 64665, 3265, 5239, 315, 264, 8866,
                                  1778, 11958, 4633, 10971, 13};
    check_case(tokenizer,
               "A cinematic close-up of a clockwork bird taking flight.",
               cinematic, sizeof(cinematic) / sizeof(cinematic[0]));

    uint32_t *ids = NULL;
    size_t count = 99;
    CHECK(h3_tokenizer_encode(tokenizer, "", 0, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == 0 && ids == NULL);
    CHECK(h3_tokenizer_encode(tokenizer, "", 1, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == 1 && ids[0] == H3_PAD_TOKEN_ID);
    h3_tokenizer_ids_free(ids);

    h3_tokenizer_free(tokenizer);
    printf("ok: %d tokenizer checks against released Qwen vocabulary\n", checks);
    return 0;
}
