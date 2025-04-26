#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

using namespace std;

using transaction_id_t = uint32_t;

class Bitmap
{
private:
    vector<uint64_t> bits;
    size_t capacity;

    inline pair<size_t, uint64_t> get_position(size_t index) const
    {
        if (index >= capacity)
            throw out_of_range("Bitmap index out of range");
        return {index >> 6, 1ULL << (index & 63)};
    }

public:
    Bitmap(size_t size = 0) : capacity(size)
    {
        if (size > 0)
        {
            size_t num_blocks = (size + 63) / 64;
            bits.resize(num_blocks, 0ULL);
        }
    }

    void set(size_t index)
    {
        if (index >= capacity)
            return;
        auto pos = get_position(index);
        bits[pos.first] |= pos.second;
    }

    int count() const
    {
        int total = 0;
        for (uint64_t block : bits)
        {
#if defined(__GNUC__) || defined(__clang__)
            total += __builtin_popcountll(block);
#else
            while (block > 0)
            {
                block &= (block - 1);
                total++;
            }
#endif
        }
        return total;
    }

    void intersect(const Bitmap &other)
    {
        if (capacity != other.capacity)
            throw invalid_argument("Bitmap size mismatch");
        size_t block_count = bits.size();
        if (block_count != other.bits.size())
            throw runtime_error("Bitmap block sizes differ unexpectedly");
        for (size_t i = 0; i < block_count; i++)
        {
            bits[i] &= other.bits[i];
        }
    }

    size_t size() const { return capacity; }

    Bitmap(Bitmap &&other) noexcept = default;
    Bitmap &operator=(Bitmap &&other) noexcept = default;
    Bitmap(const Bitmap &other) = default;
    Bitmap &operator=(const Bitmap &other) = default;
};

struct ItemBitmap
{
    int item;
    Bitmap transactions;

    ItemBitmap(int id, Bitmap b) : item(id), transactions(move(b)) {}
    ItemBitmap() : item(-1), transactions(0) {}
    bool operator<(const ItemBitmap &other) const { return item < other.item; }
};

struct FrequentItemset
{
    vector<int> items;
    int support;
    FrequentItemset(vector<int> is, int sc) : items(move(is)), support(sc) {}
};

int TRANSACTION_COUNT = 0;
int MIN_SUPPORT = 0;
const int MAX_ITEM_ID = 999;

void find_frequent_itemsets(const vector<int> &base_items, const vector<ItemBitmap> &candidates, vector<FrequentItemset> &results);
string format_itemset(const vector<int> &itemset);

string format_itemset(const vector<int> &itemset)
{
    vector<int> sorted = itemset;
    sort(sorted.begin(), sorted.end());
    stringstream ss;
    for (size_t i = 0; i < sorted.size(); i++)
    {
        ss << sorted[i] << (i == sorted.size() - 1 ? "" : ",");
    }
    return ss.str();
}

void find_frequent_itemsets(
    const vector<int> &base_items,
    const vector<ItemBitmap> &candidates,
    vector<FrequentItemset> &results)
{
#pragma omp parallel
    {
        vector<FrequentItemset> local_results;
#pragma omp for schedule(dynamic) nowait
        for (size_t i = 0; i < candidates.size(); i++)
        {
            const auto &current = candidates[i];
            vector<ItemBitmap> next_candidates;

            for (size_t j = i + 1; j < candidates.size(); j++)
            {
                const auto &next = candidates[j];

                Bitmap combined = current.transactions;
                combined.intersect(next.transactions);

                int support = combined.count();

                if (support >= MIN_SUPPORT)
                {
                    vector<int> new_itemset = base_items;
                    new_itemset.push_back(current.item);
                    new_itemset.push_back(next.item);
                    local_results.emplace_back(new_itemset, support);

                    next_candidates.emplace_back(next.item, move(combined));
                }
            }

            if (!next_candidates.empty())
            {
                vector<int> extended_base = base_items;
                extended_base.push_back(current.item);
                find_frequent_itemsets(extended_base, next_candidates, local_results);
            }
        }
#pragma omp critical
        {
            results.insert(results.end(),
                           make_move_iterator(local_results.begin()),
                           make_move_iterator(local_results.end()));
        }
    }
}

int main(int argc, char *argv[])
{
    double min_support_ratio = stod(argv[1]);
    string input_file = argv[2], output_file = argv[3];

    FILE *file = fopen(input_file.c_str(), "rb");
    fseek(file, 0, SEEK_END);
    size_t file_size = static_cast<size_t>(ftell(file));
    rewind(file);
    vector<char> buffer(file_size);
    size_t bytes_read = fread(buffer.data(), 1, file_size, file);
    fclose(file);

    vector<int> item_counts(MAX_ITEM_ID + 1, 0);
    TRANSACTION_COUNT = 0;
    transaction_id_t last_transaction = 0;

    if (bytes_read > 0)
    {
        transaction_id_t transaction_id = 0;
        int current_item = 0;
        bool parsing_item = false;

        for (size_t i = 0; i < bytes_read; i++)
        {
            char c = buffer[i];
            if (c >= '0' && c <= '9')
            {
                current_item = current_item * 10 + (c - '0');
                parsing_item = true;
            }
            else if (c == ',' || c == '\n')
            {
                if (parsing_item)
                {
                    if (current_item >= 0 && current_item <= MAX_ITEM_ID)
                    {
                        item_counts[current_item]++;
                    }
                }
                current_item = 0;
                parsing_item = false;
                if (c == '\n')
                {
                    transaction_id++;
                }
            }
            else if (c == '\r')
            {
                continue;
            }
        }

        if (parsing_item)
        {
            if (current_item >= 0 && current_item <= MAX_ITEM_ID)
                item_counts[current_item]++;
        }

        last_transaction = transaction_id;
        bool ends_with_newline = (bytes_read > 0 && buffer[bytes_read - 1] == '\n');
        TRANSACTION_COUNT = static_cast<int>(ends_with_newline ? last_transaction : last_transaction + 1);
    }

    if (TRANSACTION_COUNT == 0)
    {
        ofstream out(output_file);
        out.close();
        return 0;
    }

    MIN_SUPPORT = static_cast<int>(ceil(min_support_ratio * TRANSACTION_COUNT - 1e-9));
    if (min_support_ratio > 0 && MIN_SUPPORT < 1)
        MIN_SUPPORT = 1;
    if (min_support_ratio == 0.0)
        MIN_SUPPORT = 0;

    vector<bool> is_frequent(MAX_ITEM_ID + 1, false);
    vector<FrequentItemset> all_itemsets;
    vector<int> frequent_items;

    for (int item = 0; item <= MAX_ITEM_ID; item++)
    {
        if (item_counts[item] >= MIN_SUPPORT)
        {
            is_frequent[item] = true;
            all_itemsets.emplace_back(vector<int>{item}, item_counts[item]);
            frequent_items.push_back(item);
        }
    }
    item_counts.clear();

    map<int, Bitmap> item_bitmaps;
    for (int item : frequent_items)
    {
        item_bitmaps.emplace(item, Bitmap(TRANSACTION_COUNT));
    }

    if (bytes_read > 0)
    {
        transaction_id_t transaction_id = 0;
        int current_item = 0;
        bool parsing_item = false;

        for (size_t i = 0; i < bytes_read; i++)
        {
            char c = buffer[i];
            if (c >= '0' && c <= '9')
            {
                current_item = current_item * 10 + (c - '0');
                parsing_item = true;
            }
            else if (c == ',' || c == '\n')
            {
                if (parsing_item)
                {
                    if (current_item >= 0 && current_item <= MAX_ITEM_ID && is_frequent[current_item])
                    {
                        item_bitmaps.at(current_item).set(transaction_id);
                    }
                }
                current_item = 0;
                parsing_item = false;
                if (c == '\n')
                {
                    transaction_id++;
                }
            }
            else if (c == '\r')
            {
                continue;
            }
        }

        if (parsing_item)
        {
            if (current_item >= 0 && current_item <= MAX_ITEM_ID && is_frequent[current_item])
            {
                item_bitmaps.at(current_item).set(transaction_id);
            }
        }
    }

    buffer.clear();
    buffer.shrink_to_fit();
    is_frequent.clear();
    is_frequent.shrink_to_fit();

    vector<ItemBitmap> frequent_itemsets;
    for (int item : frequent_items)
    {
        auto it = item_bitmaps.find(item);
        if (it != item_bitmaps.end())
        {
            frequent_itemsets.emplace_back(item, move(it->second));
        }
    }
    item_bitmaps.clear();

    vector<FrequentItemset> larger_itemsets;
#pragma omp parallel
    {
        vector<FrequentItemset> thread_results;
#pragma omp for schedule(dynamic) nowait
        for (size_t i = 0; i < frequent_itemsets.size(); i++)
        {
            const auto &current = frequent_itemsets[i];
            vector<ItemBitmap> next_candidates;

            for (size_t j = i + 1; j < frequent_itemsets.size(); j++)
            {
                const auto &next = frequent_itemsets[j];

                Bitmap combined = current.transactions;
                combined.intersect(next.transactions);

                int support = combined.count();

                if (support >= MIN_SUPPORT)
                {
                    thread_results.emplace_back(vector<int>{current.item, next.item}, support);
                    next_candidates.emplace_back(next.item, move(combined));
                }
            }

            if (!next_candidates.empty())
            {
                vector<int> base = {current.item};
                find_frequent_itemsets(base, next_candidates, thread_results);
            }
        }
#pragma omp critical
        {
            larger_itemsets.insert(larger_itemsets.end(),
                                   make_move_iterator(thread_results.begin()),
                                   make_move_iterator(thread_results.end()));
        }
    }

    all_itemsets.insert(all_itemsets.end(),
                        make_move_iterator(larger_itemsets.begin()),
                        make_move_iterator(larger_itemsets.end()));

    ofstream out(output_file);
    out << fixed << setprecision(4);
    for (const auto &result : all_itemsets)
    {
        out << format_itemset(result.items) << ":"
            << static_cast<double>(result.support) / TRANSACTION_COUNT
            << "\n";
    }
    out.close();

    return 0;
}