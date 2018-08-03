#include <cmath>
#include <fstream>
#include <iostream>
#include <GL/glut.h>
#include <vector>

using namespace std;

class ObjLoader
{
public:
	ObjLoader(string filename);
	void Draw();
private:
	vector<vector<GLfloat>>vSets;//存放顶点(x,y,z)坐标
	vector<vector<GLint>>fSets;//存放面的三个顶点索引
};