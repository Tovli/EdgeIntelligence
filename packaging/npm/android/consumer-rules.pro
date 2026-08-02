# UniFFI's React Native runtime uses JNA for native symbol access. Preserve it
# when a consuming app enables R8/ProGuard.
-dontwarn java.awt.**
-keep class com.sun.jna.** { *; }
-keepclassmembers class * extends com.sun.jna.** { public *; }
