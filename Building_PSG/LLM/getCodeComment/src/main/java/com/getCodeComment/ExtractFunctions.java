package com.getCodeComment;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;

import java.util.stream.Collectors;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Optional;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.lang.String;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


// mvn compile
// mvn exec:java

public class ExtractFunctions {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("用法: java ExtractFunctions <inputFilePath> <outputRootDirectory>");
            System.exit(1);
        }

        String inputFilePath = args[0];
        String outputRootDirectory = args[1];

        System.out.println("Start!");
        int cnt = 0;
        AtomicLong processedLines = new AtomicLong(0);

        // 使用固定大小的线程池
        // int numThreads = 4; // 获取可用的处理器核心数
        // ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath))) {
            long totalLines = Files.lines(Paths.get(inputFilePath)).count();
            String line;

            while ((line = reader.readLine()) != null) {
                final String currentLine = line; // 声明为 final 或者实际上的最终变量
                cnt += 1;
                String[] parts = line.split("/");
                processedLines.incrementAndGet();
                printProgressBar(processedLines.get(), totalLines);

                //Path path = Paths.get("/mnt/inside_15T/PPG_dataset/DeUEDroid_decompile/Porn/"+parts[6]);
                // 检查文件是否存在
                //boolean directoryExists = Files.exists(path) && Files.isDirectory(path);
                //System.out.println(currentLine);
                processAllJavaFile(currentLine, outputRootDirectory, parts[9]); 
                // processAllJavaFile(currentLine, outputRootDirectory, parts[6]);
                // 提交任务给线程池
                // executorService.submit(() -> processAllJavaFile(currentLine, outputRootDirectory, parts[6]));
            }

            // 等待所有任务完成
            // executorService.shutdown();
            // while (!executorService.isTerminated()) {
            //     // 等待所有任务完成
            // }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(cnt);
        System.out.println("End!");
    }


    private static void printProgressBar(long current, long total) {
        int progressBarLength = 50; // 进度条长度
        int progress = (int) (current * progressBarLength / total);

        System.out.print("\r[");
        for (int i = 0; i < progressBarLength; i++) {
            if (i < progress) {
                System.out.print("=");
            } else {
                System.out.print(" ");
            }
        }
        System.out.print("] " + current + "/" + total + " (" + (100 * current / total) + "%)");
    }


    private static void processJavaFile(String javaFilePath, String outputRootDirectory) {
        try {
            java.io.FileInputStream fileInputStream = new java.io.FileInputStream(javaFilePath);
            ParseResult<CompilationUnit> result = new JavaParser().parse(fileInputStream);
            CompilationUnit compilationUnit = result.getResult().orElse(null);
            if (compilationUnit != null) {
                compilationUnit.findAll(MethodDeclaration.class).forEach(method -> {
                    if (method.hasJavaDocComment() || hasComment(method)) {
                        Optional<JavadocComment> javadocComment = method.getJavadocComment();
                        StringBuilder docComment = new StringBuilder();
                        javadocComment.ifPresent(comment ->{
                            docComment.append(comment.toString());
                        });
                        String finalDocComment = docComment.toString();
                        String code = method.toString();
                        String comment = method.removeBody().toString();
                        String className = method.findAncestor(ClassOrInterfaceDeclaration.class).map(ClassOrInterfaceDeclaration::getNameAsString).orElse("UnknownClass");
                        System.out.println(className);
                        String javaFileName = Paths.get(javaFilePath).getFileName().toString();
                        String functionName = method.getName().asString();
                        String outputDirectory = Paths.get(outputRootDirectory, javaFileName).toString();
                        String functionFilePath = Paths.get(outputDirectory, className + "." +functionName + ".txt").toString();
                        String functionResult = "源代码: \n" + code + "\n" + "文档: \n" + finalDocComment + "\n" + "注释: \n" + comment;
                        saveResultToFile(functionResult, functionFilePath);
                    }
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void processAllJavaFile(String javaFilePath, String outputRootDirectory, String apkFilePath) {
        try {
            java.io.FileInputStream fileInputStream = new java.io.FileInputStream(javaFilePath);
            ParseResult<CompilationUnit> result = new JavaParser().parse(fileInputStream);
            CompilationUnit compilationUnit = result.getResult().orElse(null);
            if (compilationUnit != null) {
                compilationUnit.findAll(MethodDeclaration.class).forEach(method -> {
                    String code = method.toString();
                    String javaFileName = Paths.get(javaFilePath).getFileName().toString();
                    String functionName = method.getName().asString();
                    List<String> parameterTypes = method.getParameters().stream()
                        .map(parameter -> parameter.getType().asString())
                        .collect(Collectors.toList());
                    String parameterTypesString = String.join(", ", parameterTypes);
                    //System.out.println(parameterTypesString);
                    String outputDirectory = Paths.get(outputRootDirectory, apkFilePath.replace(".apk", "")).toString();
                    String functionFilePath = Paths.get(outputDirectory, javaFilePath.replace(".java", "").replace("/", ".") + "." + functionName + "##" + parameterTypesString).toString();
                    //System.out.println(functionFilePath);
                    List<String> functionFilePaths = Arrays.asList(functionFilePath.split("source."));
                    List<String> functionFilePaths1 = Arrays.asList(functionFilePath.split("/"));
                    String finalDirectory = functionFilePaths.get(1);
                    String finalPath = functionFilePaths1.subList(0, functionFilePaths1.size()-1)
                                   .stream()
                                   .collect(Collectors.joining("/"));
                    // System.out.println("#########");
                    // System.out.println(finalPath+"/"+finalDirectory);
                    // System.out.println("#########");
                    String functionResult = code + "\n";
                    // System.out.println(finalPath+"/"+finalDirectory);
                    saveResultToFile(functionResult, finalPath+"/"+finalDirectory);
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static boolean hasComment(MethodDeclaration method) {
        Comment comment = method.getComment().orElse(null);
        if (comment != null){
            return true;
        }
        return false;
    }

    private static void saveResultToFile(String result, String functionFilePath) {
        try {
            Files.createDirectories(Paths.get(functionFilePath).getParent());
            if(functionFilePath.length() > 255){
                functionFilePath = functionFilePath.split("##")[0]+"##";
                // System.out.println(functionFilePath);
            }
            try (FileWriter writer = new FileWriter(functionFilePath)) {
                writer.write(result);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
