#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

struct TrieNode {
    char c;
    vector<TrieNode*> children;
    int count;
};

struct MergeTable {
    unordered_map<pair<char*,char*>, int> merges;
};

