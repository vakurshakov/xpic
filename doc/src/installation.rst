Установка и сборка
==================

Сборка проекта начинается с клонирования репозитория проекта:

.. code-block:: shell

  git clone https://github.com/vakurshakov/xpic.git ${XPIC_ROOT}; cd ${XPIC_ROOT}

Переменная ``${XPIC_ROOT}`` -- корень репозитория xpic. Далее в этом разделе описаны необходимые для сборки проекта зависимости и коротко описаны процессы их установки. Также упомянуто для чего эти зависимости необходимы внутри проекта. Детальное описание работы библиотек есть в их документации, соответствующие ссылки приведены ниже.


JSON (JavaScript Object Notation) for Modern C++
------------------------------------------------

При работе с кодом удобно разделять его на компилируемую, *статическую*, часть и часто изменяемую часть -- *конфигурацию*. Конфигурацией могут быть геометрические параметры системы, параметры используемых макрочастиц, дополнительные операции вне основного вычислительного цикла (*команды*) и *диагностики*. Конфигурация программы осуществляется с помощью текстового файла формата JSON, для унификации работы с аргументами и их типами чтение файлов производится с помощью библиотеки "JSON for Modern C++".

.. rst-class:: nocolorrows

============  =========================================
Git           https://github.com/nlohmann/json
Примеры       https://github.com/nlohmann/json#examples
Документация  https://json.nlohmann.me/ 
============  =========================================

Библиотека состоит из заголовочных файлов, для установки необходимо только скачать репозиторий:

.. code-block:: shell

  git clone https://github.com/nlohmann/json.git ./external/json


MPI (Message Passing Interface)
-------------------------------

MPI -- это стандарт для организации параллельных вычислений на кластерах и суперкомпьютерах. Он позволяет разным процессам обмениваться данными, синхронизировать свою работу и благодаря нему можно распределять нагрузку между множеством вычислительных узлов. Есть несколько реализаций этого стандарта и для разработки приложения выбор конкретного не важен, поскольку зачастую мы пользуемся сторонней библиотекой использующей в свою очередь MPI (о самой библиотеке ниже); использование MPI непосредственно регламентируется стандартом.

OpenMPI -- одна из популярных и производительных реализаций стандарта MPI, она используется на вычислительном кластере, поэтому привожу информацию о ней. 

.. rst-class:: nocolorrows

============  ==================================================================
Сайт          https://www.open-mpi.org/
Исходники     https://www.open-mpi.org/software/ompi/v5.0/
Документация  https://docs.open-mpi.org/en/v5.0.x/index.html
Установка     https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/index.html
Материалы     https://parallel.ru/tech/tech_dev/mpi.html
============  ==================================================================


PETSc (Portable, Extensible Toolkit for Scientific Computation)
---------------------------------------------------------------

Для работы с векторами и матрицами, решения линейных и нелинейных векторных уравнений используется библиотека PETSc, написанная на языке C. Библиотека заточена для распараллеливания с помощью MPI, распараллеливание внутри отдельных процессов обеспечивается более низкоуровневыми библиотеками вроде `BLAS <https://ru.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ и `LAPACK <https://ru.wikipedia.org/wiki/LAPACK>`_, также PETSc имеет возможность гибкой настройки работы из командной строки.

.. rst-class:: nocolorrows

=========  ==========================================
Git        https://gitlab.com/petsc/petsc
Установка  https://petsc.org/release/install/
Настройка  https://petsc.org/release/install/install/
Мануал     https://petsc.org/release/manual/
Примеры    https://petsc.org/release/tutorials/
C API      https://petsc.org/release/manualpages/
=========  ==========================================

Библиотека массивная и поэтому компилируется отдельно, подключается стандартным образом: в заголовочных файлах объявляются необходимые символы, а реализация линкуется к нашему исполняемому файлу. Для установки библиотеки её необходимо скачать 

.. code-block:: shell

  git clone -b release https://gitlab.com/petsc/petsc.git ./external/petsc

Затем настроить её перед компиляцией: указать путь до ``${MPI_DIR}``, указать пути до библиотек BLAS/LAPACK, описать дополнительные параметры, -- подробности настройки можно найти по ссылке выше, либо с помощью команды

.. code-block:: shell

  cd ${XPIC_ROOT}/external/petsc; ./configure --help

Ниже приведены команды, которые настраивают, компилируют и проверяют две версии библиотеки: *дебаг*-версию, с дополнительными символами для отладки программы, и оптимизированную, *релизную*, версию, -- устанавливаются они в директории ``${PETSC_ARCH}``, данный путь затем используется при линковкe xpic и PETSc (см. ``${XPIC_ROOT}/CMakeLists.txt``).

.. code-block:: shell

  ./configure PETSC_ARCH=linux-mpi-debug \
    --with-fc=0 \
    --with-mpi-dir=${MPI_DIR} \
    --download-f2cblaslapack \
    --with-openmp=1 \
    --with-threadsafety=1 \
    ---with-openmp-kernels=true; \
  make PETSC_ARCH=linux-mpi-debug all; \
  make PETSC_ARCH=linux-mpi-debug check

.. code-block:: shell

  ./configure PETSC_ARCH=linux-mpi-opt \
    --with-fc=0 \
    --with-mpi-dir=${MPI_DIR} \
    --download-f2cblaslapack \
    --with-openmp=1 \
    --with-threadsafety=1 \
    ---with-openmp-kernels=true \
    --with-debugging=0 \
    COPTFLAGS='-O3 -march=native -mtune=native' \
    CXXOPTFLAGS='-O3 -march=native -mtune=native'; \
  make PETSC_ARCH=linux-mpi-opt all; \
  make PETSC_ARCH=linux-mpi-opt check

При настройке можно использовать предварительно загруженные пакеты, которые, например, будут использоваться для библиотек BLAS/LAPACK. Для этого, после загрузки tar-файла можно указать путь к этому файлу, чтобы он автоматически установился при сборке PETSc. Например, можно загрузить пакет ``f2cblaslapack`` локально и указать параметр конфигурации ``--download-f2cblaslapack=/path/to/f2cblaslapack-X.Y.Z.q.tar.gz``.
