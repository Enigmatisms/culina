/**
 * Implementation of Least Recently Used
 * Cache updating mechanism
 * 
 * LFU needs a map (or array) of doubly-linked queue
*/

#include <vector>
#include <iostream>
#include <unordered_map>
#include "lru.cc"

class LocalLRUCache {
private:
    DLinkNode head;
    DLinkNode tail;
public:
    void push_front(DLinkNode* node) {
        node->next = head.next;
        node->prev = &head;
        head.next->prev = node;
        head.next  = node;
    }
    
    int pop_back() {    // pop and return key (to remove the item from the hash map)
        auto back_node = tail.prev;
        back_node->prev->next = &tail;
        tail.prev = back_node->prev;
        return back_node->key;
    }

    bool valid() const noexcept {
        return head.next != &tail;
    }
};

class LFUCache {
private:
    std::vector<LocalLRUCache> lrus;
    std::unordered_map<int, std::pair<int, DLinkNode*>> mapping;
    int capacity;
public:
    LFUCache(int cap): capacity(cap) {
        lrus.reserve(8);
    }

    int get(int key) {
        auto it = mapping.find(key);
        if (it != mapping.end()) {
            int use_cnt = ++it->second.first;
            auto node   = it->second.second;

            node->prev->next = node->next;
            node->next->prev = node->prev;

            if (use_cnt >= lrus.size())
                lrus.emplace_back();
            lrus[use_cnt].push_front(node);
            return node->value;
        }
        return -1;
    }

    void put(int key, int value) {
        auto it = mapping.find(key);
        if (it != mapping.end()) {
            int use_cnt = ++it->second.first;
            auto node    = it->second.second;
            node->value  = value;

            node->prev->next = node->next;
            node->next->prev = node->prev;

            if (use_cnt >= lrus.size())
                lrus.emplace_back();
            lrus[use_cnt].push_front(node);
        } else {
            // check if the capacity is reached
            if (mapping.size() >= capacity) {
                for (auto& lru: lrus) {
                    if (lru.valid() == false) continue;
                    int del_key = lru.pop_back();
                    mapping.erase(del_key);
                    break;
                }
            }
            if (lrus.empty())
                lrus.emplace_back();
            DLinkNode* node = new DLinkNode(key, value);
            lrus[0].push_front(node);
            mapping[key] = std::pair<int, DLinkNode*>(1, std::move(node));
        }
    }
};

