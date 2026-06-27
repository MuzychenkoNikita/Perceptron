#include <iostream>
#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iostream>

#include "interface.h"
#include "matrix.h"
#include "mnist.h"
#include "network.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    //glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(2000, 2000, "Perceptron", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    InitiateInterface(window);

    Network net;
    net.load("../trained_model.h5");


    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (!isTraining) {
            Matrix Toutput = net.forward(canvasMatrix);
            float best = 0;
            for (int j = 0; j < 10; j++) {
                predictions[j] = Toutput.at(j, 0);
                if (Toutput.at(j, 0) > best) {
                    best = Toutput.at(j, 0);
                    answer = j;
                }
            }
        }



        RenderInterface(net);
        CleanInterface();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    DestroyInterface();

    glfwTerminate();


















    /*
    std::vector<Matrix> images = load_images("../dataset/train-images.idx3-ubyte");
    std::vector<uint8_t> labels = load_labels("../dataset/train-labels.idx1-ubyte");

    std::cout << (int)labels.back() << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (images.back().at(i * 28 + j, 0) > 0.5f ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }
    */

    //Matrix output = net.forward(images[0]);
    /*
    for (int i = 0; i < output.rows(); i++) {
        std::cout << output.at(i, 0) << std::endl;
        std::cout << std::endl;
    }


    for (int i = 0; i < images.size(); i++) {
        Matrix output = net.forward(images[i]);
        net.backward(images[i], output, labels[i]);

        if (i % 1000 == 0) {
            std::cout << "Loss: " << loss(output, labels[i]) << std::endl;
        }
    }
    */

    /*
    // Testing
    std::vector<Matrix> Timages = load_images("../dataset/t10k-images.idx3-ubyte");
    std::vector<uint8_t> Tlabels = load_labels("../dataset/t10k-labels.idx1-ubyte");

    net.load("../trained_model.h5");

    int correct = 0;
    for (int i = 0; i < Tlabels.size(); i++) {
        Matrix Toutput = net.forward(Timages[i]);
        int answer = 0;
        float best = 0;
        for (int j = 0; j < 10; j++) {
            if (Toutput.at(j, 0) > best) {
                best = Toutput.at(j, 0);
                answer = j;
            }
        }
        if (answer == (int)Tlabels[i]) correct++;
    }
    std::cout << " Accuracy: " << (float)correct / Tlabels.size() * 100 << "%" << std::endl;
    */

    /*
    for (int i = 0; i < Tlabels.size(); i++) {
        Matrix Toutput = net.forward(Timages[i]);
        std:: cout << "LABEL: " << (int)Tlabels[i] << std::endl;
        int answer = 0;
        float comparison = 0;
        for (int j = 0; j < Toutput.rows(); j++) {
            if (comparison < Toutput.at(j, 0)) {answer = j; comparison = Toutput.at(j, 0);}
        }
        std::cout << "ANSWER: " << answer << std::endl;
        std::cout << "============================" << std::endl;
    }
    */
    /*
    for (int epoch = 0; epoch < 5; epoch++) {
        for (int i = 0; i < images.size(); i++) {
            Matrix output = net.forward(images[i]);
            net.backward(images[i], output, labels[i]);
        }

        int correct = 0;
        for (int i = 0; i < Tlabels.size(); i++) {
            Matrix Toutput = net.forward(Timages[i]);
            int answer = 0;
            float best = 0;
            for (int j = 0; j < 10; j++) {
                if (Toutput.at(j, 0) > best) {
                    best = Toutput.at(j, 0);
                    answer = j;
                }
            }
            if (answer == (int)Tlabels[i]) correct++;
        }
        std::cout << "Epoch " << epoch + 1 << " Accuracy: " << (float)correct / Tlabels.size() * 100 << "%" << std::endl;
    }

    net.save("../trained_model.h5");
    */
    return 0;
}