#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

class Neuron
{
public:
	Neuron();
	Neuron(GLint modelLocation);
	~Neuron();

	void draw(glm::vec3 position);

private:
	GLuint VBO, EBO, VAO;
	GLint modelLoc;
};