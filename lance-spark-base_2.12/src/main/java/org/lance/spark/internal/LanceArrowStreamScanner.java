/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.spark.internal;

import org.lance.spark.LanceRuntime;
import org.lance.spark.read.LanceInputPartition;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.IOException;

/**
 * Exports a Lance fragment scan as an Arrow C Data Interface stream ({@link ArrowArrayStream}) for
 * native consumers such as Apache Gluten / Velox.
 *
 * <p>Only the {@link ArrowArrayStream} C-struct address ({@link LanceArrowStream#streamAddress()})
 * crosses the JVM/native boundary, so the consumer's Arrow build and classloader do not need to
 * match lance-spark's. All scan planning — column projection, filter pushdown, limit/offset, row-id
 * / row-address, batch size — is delegated to {@link LanceFragmentScanner}, so this path produces
 * exactly the same rows in the same order as the Spark columnar reader.
 *
 * <p>The scan is materialized one Arrow batch at a time as the consumer pulls from the stream, so
 * peak JVM Arrow memory is bounded by a single batch rather than the whole fragment.
 */
public final class LanceArrowStreamScanner {

  private LanceArrowStreamScanner() {}

  /**
   * Plans a fragment scan and exports it as an Arrow C stream.
   *
   * <p>The returned {@link LanceArrowStream} owns the exported stream and the scan backing it. Hand
   * {@link LanceArrowStream#streamAddress()} to the native consumer, let it drain the stream to
   * exhaustion, then {@link LanceArrowStream#close() close} the handle. Closing before the consumer
   * has finished reading is a use-after-free on caller-owned native memory.
   *
   * @param fragmentId the Lance fragment to scan
   * @param inputPartition the planned partition (schema, filter, limit/offset, storage options)
   * @return an open Arrow C stream handle over the fragment scan
   */
  public static LanceArrowStream export(int fragmentId, LanceInputPartition inputPartition) {
    LanceFragmentScanner fragmentScanner = LanceFragmentScanner.create(fragmentId, inputPartition);
    ArrowReader reader = null;
    ArrowArrayStream stream = ArrowArrayStream.allocateNew(LanceRuntime.allocator());
    try {
      // Double-hop over the already-planned scanner: reuse lance-spark's ArrowReader and re-export
      // it as a C stream. The export is zero-copy — the stream wraps the reader's ArrowBufs; each
      // consumer pull materializes one Arrow batch on the JVM heap, released on the next pull.
      //
      // Ownership: exportArrayStream transfers `reader` to `stream`. Releasing the stream (by the
      // consumer, or by LanceArrowStream#close) closes `reader` via the C release callback, so the
      // handle must not — and does not — close `reader` itself.
      //
      // TODO(lance#7259): once LanceScanner#exportArrowStream(long) lands upstream, replace the two
      // lines below with fragmentScanner.exportArrowStream(stream.memoryAddress()) so the Rust core
      // populates the caller's stream directly and skips the JVM-side Arrow materialization. The
      // LanceArrowStream contract below is unchanged, so no consumer needs to be touched.
      reader = fragmentScanner.getArrowReader();
      Data.exportArrayStream(LanceRuntime.allocator(), reader, stream);
    } catch (Throwable t) {
      closeQuietly(reader);
      closeQuietly(stream);
      closeQuietly(fragmentScanner);
      if (t instanceof RuntimeException) {
        throw (RuntimeException) t;
      }
      if (t instanceof Error) {
        throw (Error) t;
      }
      throw new RuntimeException(t);
    }
    return new LanceArrowStream(stream, fragmentScanner);
  }

  private static void closeQuietly(AutoCloseable closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (Exception ignore) {
        // Best effort on the construction error path.
      }
    }
  }

  /**
   * Owns an exported {@link ArrowArrayStream} together with the fragment scan behind it.
   *
   * <p>{@link #close()} releases, in order, the exported stream (which in turn closes the Arrow
   * reader it took ownership of, freeing native scan buffers) and then the Lance scanner and
   * dataset handles. Closing the stream and closing the scanner are distinct resources: the stream
   * owns the row-producing reader, while the scanner owns the open dataset.
   */
  public static final class LanceArrowStream implements AutoCloseable {
    private final ArrowArrayStream stream;
    private final LanceFragmentScanner fragmentScanner;

    LanceArrowStream(ArrowArrayStream stream, LanceFragmentScanner fragmentScanner) {
      this.stream = stream;
      this.fragmentScanner = fragmentScanner;
    }

    /** The Arrow C Data Interface stream backing this scan. */
    public ArrowArrayStream stream() {
      return stream;
    }

    /**
     * The C-struct address to hand to a native consumer (e.g. a Velox Arrow-stream source). Valid
     * until {@link #close()}.
     */
    public long streamAddress() {
      return stream.memoryAddress();
    }

    @Override
    public void close() throws IOException {
      Throwable primary = null;
      // Closing the stream releases the exported reader (and its native scan buffers) via the C
      // release callback; then release the scanner and the dataset it holds open.
      primary = closeAndAccumulate(stream, primary);
      primary = closeAndAccumulate(fragmentScanner, primary);
      if (primary != null) {
        if (primary instanceof IOException) {
          throw (IOException) primary;
        }
        if (primary instanceof RuntimeException) {
          throw (RuntimeException) primary;
        }
        if (primary instanceof Error) {
          throw (Error) primary;
        }
        throw new IOException(primary);
      }
    }

    private static Throwable closeAndAccumulate(AutoCloseable closeable, Throwable primary) {
      if (closeable == null) {
        return primary;
      }
      try {
        closeable.close();
        return primary;
      } catch (Throwable t) {
        if (primary != null) {
          primary.addSuppressed(t);
          return primary;
        }
        return t;
      }
    }
  }
}
