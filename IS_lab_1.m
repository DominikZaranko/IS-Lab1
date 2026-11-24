% Classification using perceptron

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');


%selecting features(color, roundness, 3 apples and 2 pears)
x1=[spalva_color(A1) spalva_color(A2) spalva_color(A3) spalva_color(P1) spalva_color(P2)];
x2=[apvalumas_roundness(A1) apvalumas_roundness(A2) apvalumas_roundness(A3) apvalumas_roundness(P1) apvalumas_roundness(P2)];

%Desired output vector
T=[1;1;1;-1;-1];
labels = ["Pear", "Apple"]; %-1, 1

%% train single perceptron with two inputs and one output

eta = 0.1;
e = 1e-5;

% generate random initial values
w1 = randn(1);
w2 = randn(1);
b = randn(1);

plt = [w1, w2, b, e];
%epoch = 0;
while e ~= 0
    e=0;
    for i=1:numel(T)
        v = x1(i)*w1 + x2(i)*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end

        e_n = T(i) - y;
        w1 = w1 + eta*e_n*x1(i);
        w2 = w2 + eta*e_n*x2(i);
        b = b + eta*e_n;
        e = e + abs(e_n);
    end
    %epoch = epoch + 1;
    %plt = cat(1, [w1, w2, b, e], plt);
end
%fprintf("epoch=%d, w1=%f, w2=%f, b=%f, e=%f\n", epoch,w1, w2,b, e)
%plot(plt, Marker='x')
%legend("w1", "w2", "b", "error")

%% Gaussian Naive Bayes classifier

classes = unique(T); % 1;-1

classPrior = [0,0];
classMean = zeros(2, 2);
classVariance = zeros(2, 2);

for class = 1:numel(classes)
    inClass = T == classes(class);
    classData = [x1(:, inClass); x2(:, inClass)];
    classPrior(class) = sum(inClass) / numel(T);
    classMean(class, :) = mean(classData, 2);
    classVariance(class, :) = max(var(classData, 1, 2), 1e-9);
end


%% Testing
testImages = {A4, A5, A6, A7, A8, A9, P3, P4};
testLabels = [1 1 1 1 1 1 -1 -1];

perceptronCorrect = 0;
bayesCorrect = 0;

for i = 1:numel(testImages)
    testFeatures = [spalva_color(testImages{i}), ...
                    apvalumas_roundness(testImages{i})];

    v = testFeatures(1)*w1 + testFeatures(2)*w2 + b;
    perceptronPrediction = 1;
    if v <= 0
        perceptronPrediction = -1;
    end

    score = zeros(numel(classes), 1);

    for class = 1:numel(classes)
        z = (testFeatures - classMean(class, :)) ./ ...
            sqrt(classVariance(class, :));

        gaussianProbability = prod(exp(-0.5*z.^2) ./ ...
            sqrt(2*pi*classVariance(class, :)));

        score(class) = classPrior(class) * gaussianProbability;
    end

    [~, bayesIndex] = max(score);
    bayesPrediction = classes(bayesIndex);

    perceptronCorrect = perceptronCorrect + ...
        (perceptronPrediction == testLabels(i));
    bayesCorrect = bayesCorrect + ...
        (bayesPrediction == testLabels(i));
end

perceptronAccuracy = perceptronCorrect / numel(testImages) * 100;
bayesAccuracy = bayesCorrect / numel(testImages) * 100;

fprintf("Perceptron accuracy: %.2f%%\n", perceptronAccuracy);
fprintf("Bayes accuracy: %.2f%%\n", bayesAccuracy);