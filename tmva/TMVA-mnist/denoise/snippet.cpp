#include <string>
#include <vector>
#include <sstream> 
#include <iostream> 
#include <fstream> 
  
std::vector<std::vector<double>> parse2DCsvFile(std::string inputFileName){
 
    std::vector<std::vector<double> > data;
    std::ifstream inputFile(inputFileName);
    int l = 0;
 
    while (inputFile){
        l++;
        std::string s;
        if (!std::getline(inputFile, s)) break;
        if (s[0] != '#'){
            std::istringstream ss(s);
            std::vector<double> record;
            while (ss){
                std::string line;
                if (!std::getline(ss, line, ',')) break;
                try { record.push_back(stof(line)); }
                catch (const std::invalid_argument e) {
                    std::cout << "NaN found in file " << inputFileName << " line " << l << std::endl;
                    e.what();
                }
            }
            data.push_back(record);
        }
    }
 
    if (!inputFile.eof()){
        std::cerr << "Could not read file " << inputFileName << "\n";
    }
 
    return data;
}
 
int main()
{
    //std::vector<std::vector<double>> data = parse2DCsvFile("test.txt");
 
    std::vector<double> vec(5, 1.1);
    std::vector<std::vector<float>> data2(4, std::vector<float>(6, 0.));
    for (auto l : data2) {
        for (auto x : l)
            std::cout << x << " ";
        std::cout << std::endl;
    }
    // data.size() , data[i].size()

    return 0;
}