#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

// WASD + Q/E fly camera. Hold right mouse to look, Shift to move faster.
struct FlyCamera {
    glm::vec3 position{0.0f, 2.0f, 0.0f};
    float yaw = 0.0f;   // radians, 0 looks down +Z
    float pitch = -0.1f;
    float fovYDeg = 60.0f;
    float speed = 8.0f;
    float nearZ = 0.1f;
    float farZ = 2000.0f;

    glm::vec3 forward() const {
        return {std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch)};
    }

    glm::mat4 view() const {
        return glm::lookAt(position, position + forward(), glm::vec3(0, 1, 0));
    }

    glm::mat4 projection(float aspect) const {
        return glm::perspective(glm::radians(fovYDeg), aspect, nearZ, farZ);
    }

    void update(GLFWwindow* window, float dt, bool uiWantsMouse, bool uiWantsKeyboard) {
        double x = 0.0, y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        const bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (rmb && (looking_ || !uiWantsMouse)) {
            if (looking_) {
                yaw -= float(x - lastX_) * 0.003f;
                pitch -= float(y - lastY_) * 0.003f;
                pitch = glm::clamp(pitch, -1.55f, 1.55f);
            }
            looking_ = true;
        } else {
            looking_ = false;
        }
        lastX_ = x;
        lastY_ = y;

        if (uiWantsKeyboard) return;
        const glm::vec3 f = forward();
        const glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
        glm::vec3 move(0.0f);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += r;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= r;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move.y += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move.y -= 1.0f;
        if (glm::dot(move, move) > 0.0f) {
            const bool fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
            position += glm::normalize(move) * speed * (fast ? 4.0f : 1.0f) * dt;
        }
    }

private:
    double lastX_ = 0.0, lastY_ = 0.0;
    bool looking_ = false;
};
