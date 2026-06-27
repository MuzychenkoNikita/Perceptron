//
// Created by Nikita on 6/26/2026.
//

#ifndef UNTITLED_INTERFACE_H
#define UNTITLED_INTERFACE_H

#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "matrix.h"
#include "network.h"
#include "mnist.h"

inline ImGuiIO* io = nullptr;

int canvasWidth = 28;
int canvasHeight = 28;
int pixelSize = 50;

// 2D array or flat vector to store your pixel color data
std::vector<ImU32> pixels(canvasWidth * canvasHeight, IM_COL32(255, 255, 255, 255));
Matrix canvasMatrix(784, 1);
std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
int answer = 0;
bool showTrainWindow = false;

// at the top of interface.h
#include <thread>
#include <atomic>

inline std::thread trainThread;
inline std::atomic<bool> isTraining = false;
inline std::atomic<int> trainEpoch = 0;
inline std::atomic<int> trainSample = 0;
inline std::atomic<int> totalSamples = 0;
inline std::atomic<float> accuracy = 0.f;

void TrainWindow(Network& net) {


    if (ImGui::Begin("Training")) {
    static int epoches = 1;
    ImGui::InputInt("Number of Epoches to train", &epoches, 1);
    if (epoches < 0) epoches = 0;

    ImGui::Text("Current accuracy: %.1f%%", (float)accuracy);

    if (isTraining) {
        ImGui::Text("Training epoch %d, sample %d / %d", (int)trainEpoch, (int)trainSample, (int)totalSamples);
        float progress = totalSamples > 0 ? (float)trainSample / (float)totalSamples : 0.f;
        ImGui::ProgressBar(progress, ImVec2(ImGui::GetFontSize() * 25, 0.f));
    }

    if (!isTraining) {
        if (ImGui::Button("Start Training")) {
            if (trainThread.joinable()) trainThread.join();

            int epochesCopy = epoches;
            trainThread = std::thread([&net, epochesCopy]() {
                isTraining = true;

                std::vector<Matrix> images = load_images("../dataset/train-images.idx3-ubyte");
                std::vector<uint8_t> labels = load_labels("../dataset/train-labels.idx1-ubyte");
                std::vector<Matrix> Timages = load_images("../dataset/t10k-images.idx3-ubyte");
                std::vector<uint8_t> Tlabels = load_labels("../dataset/t10k-labels.idx1-ubyte");

                totalSamples = images.size();

                for (int epoch = 0; epoch < epoches; epoch++) {
                    trainEpoch = epoch + 1;
                    for (int i = 0; i < (int)images.size(); i++) {
                        trainSample = i;
                        Matrix output = net.forward(images[i]);
                        net.backward(images[i], output, labels[i]);
                    }

                    int correct = 0;
                    for (int i = 0; i < (int)Tlabels.size(); i++) {
                        Matrix Toutput = net.forward(Timages[i]);
                        int ans = 0; float best = 0;
                        for (int j = 0; j < 10; j++) {
                            if (Toutput.at(j, 0) > best) {
                                best = Toutput.at(j, 0);
                                ans = j;
                            }
                        }
                        if (ans == (int)Tlabels[i]) correct++;
                    }
                    accuracy = (float)correct / Tlabels.size() * 100;
                    std::cout << "Epoch " << epoch + 1 << " Accuracy: " << (float)correct / Tlabels.size() * 100 << "%" << std::endl;
                }

                net.save("../trained_model.h5");

                isTraining = false;
            });
        }
    } else {
        ImGui::BeginDisabled();
        ImGui::Button("Start Training");
        ImGui::EndDisabled();
    }

} ImGui::End();
}

void CanvasWindow(Network& net) {

    if (ImGui::Begin("Canvas")) {
        if (ImGui::Button("Clear")) {
        for (int y = 0; y < canvasHeight; ++y) {
            for (int x = 0; x < canvasWidth; ++x) {
                pixels[y * canvasWidth + x] = IM_COL32(255, 255, 255, 255);
            }
        }
        std::cout << "Canvas was cleared" << std::endl;
    }

    ImGui::SameLine();
    if (ImGui::Button("Train")) {
        if (!showTrainWindow) {
            showTrainWindow = true;
        } else showTrainWindow = false;
    }
        if (showTrainWindow) TrainWindow(net);



    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImVec2(canvasWidth * pixelSize, canvasHeight * pixelSize);

    // Create an invisible button that takes up the canvas area
    ImGui::InvisibleButton("pixel_canvas", canvasSize);
    bool hovered = ImGui::IsItemHovered();
    bool active = ImGui::IsItemActive();

    // Handle drawing on click/drag
    if (active || hovered) {
        ImVec2 mousePos = ImGui::GetMousePos();
        int px = (int)((mousePos.x - canvasPos.x) / pixelSize);
        int py = (int)((mousePos.y - canvasPos.y) / pixelSize);

        if (px >= 0 && px < canvasWidth && py >= 0 && py < canvasHeight) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                pixels[py * canvasWidth + px] = IM_COL32(0, 0, 0, 255); // Draw Black
            } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                pixels[py * canvasWidth + px] = IM_COL32(255, 255, 255, 255); // Erase (White)
            }
        }
    }

    // Render the Canvas and Grid
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    for (int y = 0; y < canvasHeight; ++y) {
        for (int x = 0; x < canvasWidth; ++x) {
            ImVec2 minp = ImVec2(canvasPos.x + x * pixelSize, canvasPos.y + y * pixelSize);
            ImVec2 maxp = ImVec2(minp.x + pixelSize, minp.y + pixelSize);

            // Draw Pixel
            drawList->AddRectFilled(minp, maxp, pixels[y * canvasWidth + x]);

            // Draw Grid Lines
            drawList->AddRect(minp, maxp, IM_COL32(200, 200, 200, 100));
        }
    }

    for (int y = 0; y < canvasHeight; ++y) {
        for (int x = 0; x < canvasWidth; ++x) {
            // Black pixel = drawn = 1, White pixel = blank = 0
            float value = (pixels[y * canvasWidth + x] == IM_COL32(0, 0, 0, 255)) ? 1.0f : 0.0f;
            canvasMatrix.set(y * canvasWidth + x, 0, value);
        }
    }

    // answers
    for (int i = 0; i < 10; ++i) {
        float prediction = predictions[i];
        ImGui::SetCursorScreenPos({canvasPos.x + 1500, canvasPos.y + 100 + 100 * i});
        ImGui::Text("%d", i);
        ImGui::SetCursorScreenPos({canvasPos.x + 1550, canvasPos.y + 100 + 100 * i});
        ImGui::ProgressBar(prediction, ImVec2(ImGui::GetFontSize() * 25, 0.0f));
    }
    ImGui::SetCursorScreenPos({canvasPos.x + 1550, canvasPos.y + 1000});
    ImGui::SetWindowFontScale(40);
    ImGui::Text("%d", answer);
    ImGui::SetWindowFontScale(1);
    } ImGui::End();

}

inline void InitiateInterface(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCanvasResizeCallback("#canvas");
#endif
    ImGui_ImplOpenGL3_Init("#version 330");
}

inline void RenderInterface(Network& net) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::DockSpaceOverViewport();

    ImGui::ShowDemoWindow();
    CanvasWindow(net);
    ImGui::Render();



    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}



inline void CleanInterface() {
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (io->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

inline void DestroyInterface() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

#endif //UNTITLED_INTERFACE_H
