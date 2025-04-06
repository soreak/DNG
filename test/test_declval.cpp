#include <type_traits>
#include <iostream>

template <typename T>
void test() {
    using Type = decltype(std::declval<T>());  // ✅ 正确
    std::cout << "Type is valid.\n";
}

int main() {
    test<int>();  // 运行时不会调用 declval，因此不会报错
}
