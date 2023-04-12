#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

class Node
{
public:
	Node();
	Node(GLint modelLocation);
	~Node();

	void draw(glm::vec3 position);

private:
	GLuint VBO, EBO, VAO;
	GLint modelLoc;
};